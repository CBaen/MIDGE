# Team 04: Competitor Landscape & Fastest Implementation Path
**Expedition:** FTMO Viability for MIDGE
**Date:** 2026-03-09
**Researcher:** Opus 4.6 sub-agent

---

## Summary Verdict

The algo prop trading landscape is thin on open-source precedent and heavy on vague anecdote — but what exists points in one consistent direction: the main cause of failure is execution risk management, not signal quality. MIDGE has the better-than-average problem: its intelligence layer is more sophisticated than anything in public algos, but its execution layer does not exist yet. The gap between what MIDGE outputs (`ConvergenceAlert`) and what the FTMO engine needs (`signal / stop_loss / take_profit`) is bridgeable in a single file with approximately 100–150 lines of Python. That bridge is the critical path.

The fastest viable route to a working FTMO attempt is a 6-component execution package under `mae_core/market/execution/` that ports the sibling's engine, translates convergence alerts to the signal contract, and enforces FTMO risk rules. Estimated effort: 2–3 focused sessions. Validation: run the historical replay harness through the combined system before any live attempt.

---

## Finding 1: The Open-Source Algo Prop Trading Landscape Is Essentially Empty

### What exists (GitHub, verified 2026-03-09)

Only 6 public repositories are tagged "ftmo" on GitHub. The most relevant:

- **EA_SCALPER_XAUUSD** (53 stars, MQL5 + Python): XAUUSD trading EA for Apex and FTMO. Uses MQL5 for execution with Python-based regime detection. The public snapshot explicitly excludes "proprietary rules, parameters, or operational go-live wiring" — it's an educational shell, not a functional system. The private `aurum-pro` repository holds the real code.
- **silicon-metatrader5** (15 stars): Docker-based MT5 solution for running algorithmic trading on macOS, not a trading system itself.
- **PropForge** (9 stars, HTML): Interactive prop firm training simulator — useful for understanding rules, not for signal-to-execution.

No Python repositories were found combining FTMO + risk management + algo execution in a functional form. GitHub search for "FTMO risk management python" and "prop firm risk management python" returned zero results.

**Implication for MIDGE:** There are no open-source competitors to benchmark against or borrow from. MIDGE is building this without a reference implementation. This is a green-field gap in the public ecosystem, which confirms the sibling's assessment that the execution engine has to be built from scratch (which it already has been, in the sibling's project).

### The broader algorithmic trading landscape

The only mature signal-to-execution frameworks in Python are general-purpose:

- **Freqtrade** (crypto-focused): Uses a `stoploss` percentage parameter and optional `custom_stoploss()` callback. Strategy interface: `populate_entry_trend()` sets `enter_long` / `enter_short` columns. Exit is handled via separate `populate_exit_trend()` or stop logic. Confidence-based position sizing is not native but implementable via `custom_stake_amount()` callback. Source: freqtrade.io/en/stable/strategy-customization/, verified 2026.
- **NautilusTrader** (mentioned in EA_SCALPER_XAUUSD): High-performance Python + Rust framework. More infrastructure overhead than needed for a first integration.

Neither is directly relevant — MIDGE's integration doesn't need a full framework, it needs a translation layer. The signal contract is already defined by the sibling's engine.

---

## Finding 2: FTMO's Official Position on Algorithmic Trading

**Source:** ftmo.com/en/forbidden-trading-practices/, verified 2026-03-09

FTMO explicitly allows algorithmic trading (EAs). The forbidden practices that affect automated systems are:

### Prohibited (relevant to MIDGE)

1. **Over 2,000 server requests per day** per EA. MIDGE's sensing hook fires every 25 steps in daemon mode. If MIDGE runs at 1 step/second with 12 concurrent workers, that's far under 2,000 requests/day to any single broker API.

2. **News trading prohibition**: Cannot open positions "when major global news, macroeconomic events, or corporate reports or earnings are scheduled" or trade "two hours or less before a relevant financial market is closed for at least two hours." MIDGE already has Economic Calendar suppression windows (FOMC/CPI/NFP suppression) — this is already aligned. The signal translator must check this gate before submitting any order.

3. **"Artificially distribute profit across multiple days without proportionally distributing market risk."** This targets Martingale-style systems. MIDGE's convergence signals don't do this. Not a concern.

4. **Coordinated account abuse**: Trading connected accounts in concert. Not applicable.

5. **Exploiting price display errors or delays**: MIDGE uses standard market prices from yfinance/Polygon. Not applicable.

6. **"Strategies not in line with risk management rules a reasonable person would apply" and "unusually larger or smaller number of positions compared to other simulated trades."** This vague clause is the primary risk. It has been used (per trader community reports) to terminate consistently profitable algorithmic accounts. The mitigation is: keep position sizing conservative and consistent, avoid extreme clustering of trades on single days.

### Allowed

- EA trading on MT4, MT5, and cTrader
- Python-based trading via broker APIs (not directly stated, but not prohibited)
- Holding positions overnight
- Trading during high-volatility periods (as long as not exploiting errors)

### Platform note

FTMO supports MT4, MT5, cTrader, and DXtrade. Alpaca is a US equities/crypto broker — it is NOT a platform FTMO uses. The sibling's FTMO backtester tests the constraint logic, not the platform. For a real FTMO challenge, you need to trade FTMO's instruments (forex pairs, indices, gold, oil) via MT4/MT5 or cTrader, not via Alpaca.

**Critical gap the sibling identified but did not resolve:** Alpaca is a US equities broker. FTMO's instrument universe is forex/indices/commodities (USD/EUR, NASDAQ, Gold, Oil). These are different markets, different APIs, different order types. Alpaca paper trading is NOT a proxy for FTMO MT4/MT5 execution validation. Alpaca can validate MIDGE's signal quality against US equities — it cannot validate whether convergence signals translate to FTMO-tradeable instruments.

---

## Finding 3: What Common Algo Approaches Are Used for FTMO Challenges

Based on verified sources (team-01 findings, EA_SCALPER_XAUUSD architecture, secretstotrading101.com, 2026):

### Approaches that appear in the wild

**1. Mean Reversion (most common for challenges)**
The sibling's engine confirmed this: Bollinger Band mean reversion on AUD/USD at 2% risk achieved 75% pass rate on 250-day windows. This is consistent with community findings — mean reversion strategies fare well in prop firm challenges because they produce frequent small wins that build steadily toward the profit target without large consecutive losses. Forex and commodity mean reversion is standard for MT5 EAs.

**2. Trend following with ATR-based stops (second most common)**
EMA crossover + ATR stops. The EA_SCALPER_XAUUSD architecture (gold, regime detection) is an example. These work when a strong trend develops but produce significant drawdown during choppy markets. Less reliable for prop firm challenges than mean reversion because drawdown profiles are more volatile.

**3. News/event-driven approaches (common in FTMO context)**
Trading around scheduled economic events. MIDGE actually prohibits this via Economic Calendar suppression — which is correct behavior under FTMO's rules.

**4. Multi-signal convergence (rare, no public implementations)**
No public implementations found. MIDGE's approach — combining 12 independent domains via Thompson-weighted Bayesian convergence — is not replicated in any public repository. This is the structural moat Team-03 likely confirms academically.

### Failure patterns (what causes algo prop trading failures)

Synthesized from secretstotrading101.com, team-01 sources, earnforex.com forum reports (2024-2025):

1. **Over-risking**: The most common failure mode. Algos set to 2-5% risk per trade to hit the profit target faster, then hit one adverse streak and breach the drawdown limit. MIDGE's Kelly sizer defaults to aggressive sizing — this needs explicit capping at 1-2% for FTMO.

2. **Backtest overfitting**: Strategies are curve-fit to historical data. In live conditions, signal quality degrades. MIDGE's Bayesian Thompson sampling mitigates this by updating signal reliability on live data — a structural advantage.

3. **Insufficient signal frequency on FTMO instruments**: MIDGE's signals may primarily land on small-cap US equities (FinViz unusual volume, SEC Form 4 filings). FTMO trades forex/indices/commodities. If MIDGE's best convergence combos don't fire on FTMO-tradeable instruments, the effective signal frequency could be near zero. This is the single largest unquantified risk.

4. **Execution slippage**: MT4/MT5 slippage on news events, wide spreads on exotic pairs. Mitigation: stick to major forex pairs (EUR/USD, GBP/USD) and major indices (US30, NASDAQ100) where spreads are tightest.

5. **Best Day Rule violation**: MIDGE's convergence alerts can cluster — multiple domains confirming simultaneously creates the risk of one day generating >50% of total profit under FTMO's 1-Step Best Day Rule. The 2-Step challenge does NOT have this rule — a meaningful reason to prefer 2-Step for MIDGE.

6. **Account termination for undisclosed reasons ("exploitative practices")**: Verified by multiple 2024-2025 forum reports. Profitable algorithmic accounts terminated without clear rule violation cited. FTMO's defense: vague "exploitative" clause. This is structural risk, not preventable.

---

## Finding 4: The Signal Gap — What Needs to Be Built

### Current MIDGE output (ConvergenceAlert)

From `mae_core/market/intelligence/convergence_alerter.py`:

```python
@dataclass
class ConvergenceAlert:
    alert_id: str
    timestamp: datetime
    direction: str          # "bullish" or "bearish"
    strength: float         # 0-1 overall convergence strength
    confidence: float       # 0-1 Bayesian reliability estimate
    domains_converging: List[str]   # ["insider", "macro", "technical"]
    signals: List[Signal]   # raw signals that triggered convergence
    cross_domain_count: int
    summary: str
    urgency: str            # "immediate", "hours", "days"
    coherence: float        # 1.0 = all signals agree
    combo_key: str          # "combo:events+macro+price"
    domain_sequence: List[str]   # temporal ordering
    sequence_score: float        # 0.5-1.3 lag-relationship multiplier
    ripple_effects: List[dict]   # WorldModel downstream predictions
```

### Required engine input (FTMOBacktester.run())

From `project_cameron/trading/engine.py`:

```python
signals['signal']      # 1 = long, -1 = short, 0 = no trade
signals['stop_loss']   # absolute price level
signals['take_profit'] # absolute price level
```

### The translation bridge (what must be written)

A `SignalTranslator` that maps `ConvergenceAlert` → `(signal, stop_loss, take_profit)`:

**Step 1: Direction**
- `direction == "bullish"` → `signal = 1`
- `direction == "bearish"` → `signal = -1`
- `coherence < 0.6` → `signal = 0` (discard contradictory convergences)

**Step 2: Entry price**
- Use current market price at alert timestamp. MIDGE already has `price_fetcher.py` for this.

**Step 3: ATR lookup**
- MIDGE has vectorized ATR in `mae_core/market/edge/ta_indicators.py`. ATR period = 14.
- ATR can be computed from the same OHLCV data used for TA signals — this is free (already fetched).

**Step 4: Stop loss**
- `stop_loss = entry_price - (ATR × 1.5)` for longs
- `stop_loss = entry_price + (ATR × 1.5)` for shorts
- 1.5x ATR is the standard institutional stop distance — wide enough to avoid noise, tight enough to risk-define the trade.

**Step 5: Take profit**
- `take_profit = entry_price + (ATR × 3.0)` for longs
- `take_profit = entry_price - (ATR × 3.0)` for shorts
- 2:1 R:R at minimum. The sibling used 3x ATR in `strategies.py` for all trend strategies.

**Step 6: Position size (confidence scaling)**
- Map `confidence` to `risk_per_trade_pct`:
  - `confidence >= 0.80` → 2.0% risk
  - `confidence >= 0.65` → 1.5% risk
  - `confidence >= 0.50` → 1.0% risk
  - `confidence < 0.50` → skip (below convergence threshold)
- This replaces flat position sizing with Bayesian-informed sizing — the key differentiator from the sibling's dummy strategies.

**Step 7: News gate**
- Check MIDGE's Economic Calendar suppression windows before submitting. If a major event is within 2 hours, skip regardless of signal strength.

This is approximately 100-150 lines of Python. No new architecture required.

---

## Finding 5: Fastest Path to a Working FTMO Attempt

### Phase 1: Historical validation (can be done now, no external dependencies)

**Task:** Run MIDGE's historical convergence alerts (from `data/midge/signals/`) through the combined system:
1. Load past `ConvergenceAlert` records from the signal archive
2. Pass each through the `SignalTranslator` (to be built)
3. Feed the resulting `(signal, stop_loss, take_profit)` into the sibling's `FTMOBacktester.run()`
4. Measure simulated pass rate, drawdown profile, and time-to-pass

This answers the critical question: do MIDGE's actual historical convergence signals pass the FTMO challenge simulation?

**Constraint:** MIDGE's signals are primarily on US equities. FTMO trades forex/indices. The historical validation will need to use either:
- The underlying price data for FTMO-tradeable instruments where MIDGE fires signals (if any)
- Or substitute with SPY/QQQ as proxies for NASDAQ/US30 indices (acceptable as a first approximation)

**Effort:** 1 session to build the translator and wire the replay. This is the highest-priority task.

### Phase 2: Instrument gap audit

**Task:** Audit the last 30 days of live convergence alerts. Categorize each by instrument type:
- US equities (FinViz, SEC Form 4, OpenInsider) — NOT directly FTMO-tradeable
- Indices (SPY, QQQ, DIA) — can map to NASDAQ100/US30/US500 on FTMO
- Commodities (EIA energy signals → crude oil CL) — directly FTMO-tradeable
- Crypto (CoinGecko/CoinCap) — FTMO offers BTC/USD, ETH/USD
- Forex (FRED macro, COT positioning) — directly FTMO-tradeable if paired

**Effort:** 1 hour analysis script over the discovery log.

### Phase 3: Free trial validation (requires Guiding Light to register)

**What:** FTMO 14-day free trial at ftmo.com. Zero cost. Run the execution engine live on paper with FTMO's actual prices and MT4/MT5. This is the only way to validate the MT4/MT5 execution pathway without spending money.

**Human action required:** Guiding Light signs up at ftmo.com (browser, 10 minutes). Provides credentials to next instance.

**MT4/MT5 bridge gap:** MIDGE doesn't have an MT4/MT5 bridge — it has the Alpaca client. For the free trial, the workaround is manual: MIDGE generates alerts, a human (or a future MT4 bridge) executes them. A proper MT4/MT5 Python bridge uses MetaAPI (metatrading.io) or MetaTrader's own Python bindings — approximately 50-100 lines of integration code.

### Phase 4: First real challenge ($165)

Only proceed after Phase 1 historical validation shows positive EV and Phase 3 free trial confirms the execution pathway works.

---

## Finding 6: Platform Comparison — Alpaca vs MT4/MT5 vs cTrader

### Alpaca (what MIDGE has)

**Strengths:**
- Python-native. `AlpacaClient` already built with bracket orders.
- Excellent for US equity execution validation.
- Paper trading available immediately (zero setup).
- Bracket orders with take-profit and stop-loss supported.
- REST API + WebSocket streaming.

**Critical limitations for FTMO:**
- Alpaca trades US equities and crypto. FTMO's challenge is forex/indices/commodities.
- Cannot execute FTMO trades. FTMO requires MT4/MT5 or cTrader accounts at their designated brokers.
- Alpaca paper trading latency: standard REST API, ~50-200ms per order. This is irrelevant for MIDGE's swing-trade timescale (signals fire with "days" urgency, not milliseconds).

**Verdict:** Use Alpaca for US equity signal validation in parallel. Do not mistake Alpaca paper trading for FTMO challenge validation — they are different environments.

### MT4/MT5 (what FTMO uses)

**For automation:**
- Native: MQL4/MQL5 EA (requires writing in MetaQuotes Language — C-like, not Python)
- Python bridge options:
  - **MetaAPI** (metatrading.io): REST/WebSocket API wrapping MT4/MT5. Pricing: free tier (100K requests/month), paid tiers. This is the fastest Python→MT5 bridge.
  - **mt5linux** (GitHub): Python package for MT5 on Linux via Wine. Works but fragile.
  - **MetaTrader5 Python package** (official): Windows-only. Requires MT5 installed locally.

**Verdict:** MetaAPI is the fastest viable Python→FTMO execution bridge. Adds one more API dependency but avoids MQL5 learning curve.

### cTrader

**For automation:** cTrader Automate uses C#-based cBots. No native Python support. Third-party libraries exist but are experimental.

**Verdict:** More complex than MT4/MT5 for Python integration. Skip for the first attempt.

---

## Finding 7: Alpaca as Validation Proxy — Useful But Limited

Alpaca paper trading simulates US equity execution with realistic fill assumptions. Based on Alpaca docs (docs.alpaca.markets), verified 2026:

- Market orders fill at the next available bid/ask.
- Bracket orders are supported with `order_class=OrderClass.BRACKET` + `TakeProfitRequest` + `StopLossRequest`.
- Crypto on Alpaca: GTC and IOC only (no DAY orders). This matters if MIDGE tries to use Alpaca for crypto-based FTMO trades.
- Fractional shares supported for market orders.

**Valid uses for Alpaca paper trading:**
1. Validating that the `AlpacaClient` bracket order logic actually works end-to-end.
2. Testing signal frequency — how often does MIDGE fire signals on US equities that would execute cleanly?
3. Live paper trading US equity signals to accumulate real-world accuracy data before FTMO commitment.

**Invalid uses:**
1. As a proxy for FTMO MT4/MT5 challenge performance. Different instruments, different price feeds, different spread/slippage profiles.
2. As evidence that MIDGE can pass FTMO's challenge.

---

## Finding 8: What the PropAlphaEvalSolver Confirms

The PropAlphaEvalSolver (github.com/Prop-Alpha/PropAlphaEvalSolver, Python + Streamlit, Monte Carlo-based) treats prop trading accounts as path-dependent call options. The key insight from its methodology: it's not enough to have positive expected value — the *path* matters. A strategy can have positive EV but still have a high probability of breaching drawdown limits before reaching the profit target.

This is exactly why the sibling's finding about no time limit is so important: removing the time constraint converts "will this system reach the target before the drawdown limit?" into "will this system eventually reach the target?" — two very different questions. With infinite time and positive EV, the answer to the second question is yes (almost surely). With a 30-day time limit and the same system, it may be no.

**For MIDGE:** The Monte Carlo approach from PropAlphaEvalSolver could be applied to MIDGE's actual win rate / payoff ratio numbers from the historical replay data. This would give a probabilistic pass rate estimate more rigorous than the deterministic backtest. However, the first step is just running the historical replay through the sibling's engine — the Monte Carlo simulation is a refinement, not a prerequisite.

---

## The Component Architecture (Minimum Viable Integration)

Based on the sibling's specification (`FTMO-EXECUTION-ENGINE.md`), cross-referenced with MIDGE's actual codebase:

### `mae_core/market/execution/` — 6 files

```
mae_core/market/execution/
├── __init__.py
├── ftmo_engine.py          # Port of project_cameron/trading/engine.py
│                           # Add: multi-position support, live mode
├── ftmo_config.py          # Phase 1, Phase 2, Funded configs as dataclasses
├── signal_translator.py    # ConvergenceAlert → (signal, stop_loss, take_profit)
│                           # ATR lookup, confidence→risk mapping, news gate
├── position_sizer.py       # Thompson confidence → risk_per_trade_pct
│                           # Wraps ftmo_engine.calculate_position_size()
├── challenge_tracker.py    # State machine: challenge_start, progress, pass/fail
│                           # Publishes CH_CHALLENGE_STATUS
└── risk_guardian.py        # Real-time daily loss + total DD monitor
                            # Publishes CH_RISK_ALERT, emergency stop
```

### Triadic connections (Mae's Law 1 — no bare dyads)

**Triad A:** `SignalTranslator ↔ FTMOEngine ↔ RiskGuardian`
- Translator validates signal quality before engine sees it.
- Engine executes and reports result to RiskGuardian.
- RiskGuardian can veto subsequent signals if daily limit is approached.

**Triad B:** `PositionSizer ↔ ChallengeTracker ↔ FTMOEngine`
- Sizer queries ChallengeTracker for current P&L state (scale down if near target, scale up if early).
- Tracker receives each closed trade result from Engine.
- Engine receives final position size from Sizer.

**ConnectionRegistry entries needed:** 8 directed connections for these two triads.

### EventBus subscriptions

Subscribe: `CH_CONVERGENCE_ALERT`, `CH_CASCADE_CONFIRMED` (optional boost)
Publish: `CH_TRADE_EXECUTED`, `CH_TRADE_CLOSED`, `CH_CHALLENGE_STATUS`, `CH_RISK_ALERT`

---

## Sequenced Recommendation

Ordered by dependency and impact:

1. **Build `signal_translator.py`** first. This is the bridge. It consumes ConvergenceAlert and produces the `(signal, stop_loss, take_profit)` tuple the engine needs. ~100-150 lines.

2. **Wire historical replay validation**: Take past ConvergenceAlerts from MIDGE's signal archive, run through `signal_translator.py`, feed to `FTMOBacktester.run()`. This produces an actual expected pass rate, not a theoretical one.

3. **Audit instrument overlap**: Script over the last 30 days of convergence alerts — classify each by whether it maps to an FTMO-tradeable instrument. If <20% are FTMO-tradeable, the signal generation side needs work before the execution side matters.

4. **Build remaining execution package**: Port the engine, implement challenge tracker and risk guardian, wire into bootstrap as Layer 33 sub-component.

5. **Human gate**: Guiding Light registers for FTMO free trial (14-day, zero cost). This unlocks live paper validation.

6. **MetaAPI bridge**: If free trial shows viability, build a ~100-line MetaAPI client that submits MIDGE's translated signals to MT5. This is the last mile before a real challenge attempt.

---

## Sources

- FTMO Forbidden Trading Practices: https://ftmo.com/en/forbidden-trading-practices/ (verified 2026-03-09)
- FTMO Trading Objectives (2-Step vs 1-Step rules): https://ftmo.com/en/trading-objectives/ (verified 2026-03-09)
- Sibling's FTMO execution engine + backtest findings: `C:\Users\baenb\projects\MIDGE\FTMO-EXECUTION-ENGINE.md`
- Sibling's backtester source: `C:\Users\baenb\projects\project _cameron\trading\engine.py` + `strategies.py`
- ConvergenceAlert dataclass: `C:\Users\baenb\projects\MIDGE\mae_core\market\intelligence\convergence_alerter.py`
- AlpacaClient: `C:\Users\baenb\projects\MIDGE\mae_core\market\apis\alpaca_client.py`
- Alpaca order types documentation: https://docs.alpaca.markets/reference/postorder (verified 2026-03-09)
- Team 01 prop firm economics findings: `C:\Users\baenb\projects\MIDGE\research\expedition-ftmo-viability\team-01-prop-firm-economics.md`
- EA_SCALPER_XAUUSD (FTMO + Apex algo EA): https://github.com/francomascareloai/EA_SCALPER_XAUUSD (verified 2026-03-09)
- FTMO GitHub topic repositories (6 repos): https://github.com/topics/ftmo (verified 2026-03-09)
- PropAlphaEvalSolver (Monte Carlo prop firm simulator): https://github.com/Prop-Alpha/PropAlphaEvalSolver (verified 2026-03-09)
- SecretToTrading101 FTMO pass rate analysis: https://secretstotrading101.com/how-many-people-pass-ftmo/ (verified 2026-03-09)
- Freqtrade stop loss documentation: https://www.freqtrade.io/en/stable/stoploss/ (verified 2026-03-09)
- FundedNext algo trading policy (EAs explicitly allowed): https://help.fundednext.com/en/ (verified 2026-03-09)
- TopStep instrument restrictions (CME futures only, no VPS): Team 01 sources (verified 2026)
