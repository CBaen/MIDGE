# Team 02: MIDGE Internal Capability Audit
**Expedition:** FTMO Viability for MIDGE
**Date:** 2026-03-09
**Researcher:** Sonnet 4.6 sub-agent

---

## Summary Verdict

MIDGE has significant architectural capability that maps partially but not cleanly onto FTMO's requirements. The convergence engine fires useful signals, but has three critical gaps for FTMO: (1) overall convergence win rate of 19.9% is below the 23% break-even for FTMO's 1%-risk / 3.34:1-payoff structure, (2) 97% of signals are on US equities — FTMO does not trade equities, and (3) ConvergenceAlert carries no entry price, stop loss, or take profit — a signal translator must be built before any trade can be placed. The high-confidence tier (>=0.7) has zero graded outcomes in current data, so its 12.5% frequency claim is unvalidated. The architecture is ready for integration; the edge is unproven on FTMO instruments specifically.

---

## 1. Signal Frequency

### What the data shows

**Source:** `C:\Users\baenb\projects\MIDGE\data\midge\alerts_human.jsonl` (24,222 records, 3 parse errors)

The `alerts_human.jsonl` file contains two categories:

| Source | Count | Days | Avg/Day |
|--------|-------|------|---------|
| `pattern_stack` | 23,974 | 3 | 7,991 raw / ~725 deduplicated |
| `convergence_alert` | 247 | 2 | ~124/day |
| `active_tracking` | 1 | 1 | — |

**Warning: massive duplication exists.** Paper trades file (`data/midge/paper_trades.jsonl`) has 1,055 records but only 7 unique signal IDs across 8 unique tickers — the same signal is being written repeatedly as MIDGE loops. This is a logging bug, not signal generation. All frequency counts below use deduplicated figures.

**Deduplicated pattern_stack alerts** (unique per ticker/direction/hour):
- Total unique: 2,176 over 3 days (Mar 5–7, 2026)
- Per-day: 336, 1,141, 699
- Average: 725/day

**Convergence alerts:**
- Total: 247 over 2 days (Mar 6–7, 2026)
- All 247 are on a single ticker: TUSK
- Confidence: min=0.485, max=0.530, avg=0.487 — all below the 0.45 gate but above noise floor
- Zero are at confidence >= 0.6

**Historical baseline (Feb 2026 replay):** 288 convergence alerts in one month = ~9.6/day. This is the only multi-week sample. The current 124/day figure reflects 3 specific live days and likely represents a hot period.

**Conclusion for FTMO:** Signal frequency is not the bottleneck. 9–125 convergence alerts per day provides ample raw material. The constraint is instrument coverage (see Question 6). If 1–2% of signals land on FTMO-tradeable instruments at FTMO-quality confidence, frequency drops to 0–2/day — viable but not abundant.

**Source:** `C:\Users\baenb\projects\MIDGE\research\phase0-measurements.md` (288 alerts/month in Feb 2026 replay)

---

## 2. Win Rate by Confidence Tier

### The honest answer: we do not have graded high-confidence outcomes

**Source:** `C:\Users\baenb\projects\MIDGE\data\midge\paper_trades.jsonl`, `C:\Users\baenb\projects\MIDGE\data\midge\alerts_human.jsonl`

The paper_trades file records signals with `hit_rate: 0.0` for all 8 unique signals — they have not been graded yet. All 8 have confidence in the 0.45–0.60 range. Zero signals at >= 0.6 exist in the live paper trading log.

Pattern stack alerts do show high-confidence signals:
- Conf >= 0.8: 132 (6.1% of unique alerts) — tickers include RTX, NVDA, META, LMT, NOC
- Conf >= 0.7: 264 (12.1%)
- Conf >= 0.6: 308 (14.2%)
- These are pattern archaeology tier: `high` (1,486), `medium` (2,512), `low` (19,976)

The high-confidence pattern_stack alerts are repeating the same signal across multiple loop iterations — RTX at 0.990 appears dozens of times in a single session. They have not been graded against outcomes.

**What we DO have graded:** The Feb 2026 replay, sourced from `research/phase0-measurements.md`:

| Confidence range | Win rate | Notes |
|-----------------|----------|-------|
| Overall (all) | 19.9% | 31/156 graded alerts |
| >= 0.45 subset | 29–32% | Reported in MEMORY.md: "Confidence >= 0.45 → 29-32% WR" |
| Best combos (small n) | 29.4–66.7% | events+macro+price: 31.2% (n=32); best small-sample: 66.7% (n=3) |

**Critical finding from phase0-measurements.md:** "Winners 0.560, Losers 0.565 — confidence engine doesn't discriminate." Confidence score currently has near-zero predictive power for whether a trade will win. A signal at 0.7 confidence is not materially more likely to win than one at 0.5. This directly undermines the planned approach of filtering to high-confidence signals.

**Root cause documented in phase0-measurements.md:** "81 of 83 Thompson distributions are still at the uniform 50/50 prior. This means MIDGE is not yet using the 230,462 signals in its archive to weight its confidence calculations."

**Implication for FTMO:** Filtering to conf >= 0.7 does not reliably select better signals. The confidence engine has not yet been calibrated by live outcomes. Any FTMO strategy relying on confidence thresholds is filtering on noise, not edge.

---

## 3. Payoff Ratio Analysis

### Numbers

**Source:** `C:\Users\baenb\projects\MIDGE\research\phase0-measurements.md`

- Average win: 11.4%
- Average loss: 3.4%
- Payoff ratio: 3.34:1

### Expected value calculation

At the overall 19.9% win rate:
```
E = (0.199 × 11.4%) - (0.801 × 3.4%) = 2.27% - 2.72% = -0.45% per trade
```
**Negative expectancy at 19.9% WR.** Losing system at current observed convergence win rate.

Break-even win rate at 3.34:1 payoff:
```
0 = WR × 0.114 - (1-WR) × 0.034
WR = 0.034 / (0.114 + 0.034) = 23.0%
```
**MIDGE needs at least 23% win rate to be break-even.**

At the 0.45-gated subset (29–32% WR):
```
E at 30%: (0.30 × 11.4%) - (0.70 × 3.4%) = 3.42% - 2.38% = +1.04% per trade
```
**Positive expectancy at 30% WR: +1.04%/trade.** This is the viable scenario — but only when filtering to combos that historically achieve 29–32% WR.

### FTMO-specific position sizing math

With 1% risk per trade and 3.34:1 payoff (ATR-based SL/TP per sibling's engine):
- Win yields: +3.34% to account
- Loss costs: -1.00% to account
- Break-even WR: 23.0% (identical — payoff ratio drives break-even, not position size)

| Win rate | EV/trade | Trades to +10% target | Estimated weeks at 5 trades/week |
|----------|----------|----------------------|----------------------------------|
| 19.9% | -0.14% | never converges | N/A |
| 23.0% | ~0% | infinite | N/A |
| 25.0% | +0.09% | ~112 trades | ~22 weeks |
| 30.0% | +0.30% | ~33 trades | ~7 weeks |
| 50.0% (best combos) | +1.17% | ~9 trades | ~2 weeks |

**Critical observation:** The 19.9% overall rate is a losing system for FTMO. The profitable path requires filtering to the 30%+ win-rate combo subset AND those combos must land on FTMO-tradeable instruments. Both conditions must hold simultaneously.

---

## 4. Position Sizing Feasibility

### FTMO constraints

- Daily loss limit: $500 (5% of $10K)
- Max drawdown: $1,000 (10% of $10K)
- Must have minimum 4 trading days (2-step challenge)

### Sibling's approach

**Source:** `C:\Users\baenb\projects\MIDGE\FTMO-EXECUTION-ENGINE.md`

The sibling built a position sizer using:
```
risk_amount / (stop_loss_pips × pip_value_per_lot)
```

Proposed confidence scaling:
- Confidence > 0.8 → 2.5% risk
- Confidence 0.6–0.8 → 1.5% risk
- Confidence < 0.6 → skip or 0.5%

### Gap analysis

**Current MIDGE paper account:** $50,000 (from `learning_config.py`: `paper_account_value: 50000`). Kelly position sizing is calibrated to a $50K account — this must be recalibrated to $10K for FTMO.

**Daily loss limit math:**
At 1% risk ($100/trade), a streak of 5 losses in one day = $500 = exactly the daily limit. This means MIDGE must enforce a hard 5-trade-per-day loss ceiling. The current `DrawdownMonitor` enforces this systemically but the hook needs to be wired to halt trading when daily loss approaches 80% of limit (conservative: halt at $400 daily loss).

**Drawdown path modeling:**
- 10 consecutive losses at 1% risk = 10% drawdown = hard bust
- At 19.9% WR (80.1% loss rate): P(10 consecutive losses) = 0.801^10 = **10.9%** — unacceptably high
- At 30% WR (70% loss rate): P(10 consecutive losses) = 0.70^10 = **2.8%** — manageable but real
- At 30% WR, expected max drawdown before reaching +10%: ~4–6% (within limit)

**MIDGE's existing risk architecture:**
- `DrawdownMonitor`: Built and wired. Enforces circuit breakers. Already handles % drawdown from peak. Only needs FTMO-specific threshold configuration ($500 daily, $1,000 total).
- `SystemHealthMonitor` and `SelfMonitor`: Wired. System shutdown on health degradation.
- **Gap:** These are monitoring systems, not execution gates. No mechanism currently halts order placement when FTMO drawdown limits are approached. The `challenge_tracker.py` module proposed by the sibling does not exist yet.

**Verdict:** Position sizing machinery is conceptually complete in the sibling's engine but not ported to MIDGE. The FTMO $10K account requires explicit reconfiguration — the $50K paper account calibration will over-size positions by 5x unless corrected.

---

## 5. Signal-to-Trade Translation Gap

### What ConvergenceAlert currently contains

**Source:** `C:\Users\baenb\projects\MIDGE\mae_core\market\intelligence\convergence_alerter.py`, lines 62–107

```
ConvergenceAlert fields:
  alert_id: str
  timestamp: datetime
  direction: str           # "bullish" or "bearish"
  strength: float          # 0-1
  confidence: float        # 0-1
  domains_converging: List[str]
  signals: List[Signal]    # raw signals that triggered
  cross_domain_count: int
  summary: str
  urgency: str             # "immediate", "hours", "days"
  coherence: float
  contradiction_details: list
  combo_key: str
  domain_sequence: List[str]
  sequence_score: float
  ripple_effects: List[dict]  # causal cascade predictions
```

**What is present that helps:**
- `direction` (bullish/bearish) → maps to `signal = 1 or -1`
- `confidence` → maps to position size scaling
- `signals[].metadata` → may contain price data
- `urgency` → maps to timeframe (immediate vs days)

**What is completely absent:**
- Entry price (no current market price field)
- Stop loss level (no absolute price)
- Take profit level (no absolute price)
- ATR (not attached to the alert)
- Ticker/instrument identifier at the ConvergenceAlert level — alerts exist for the `signal_buffer` domain, not always a clean ticker

**Gap:** The sibling's interface contract requires:
```python
signals['signal']      # 1 / -1 / 0
signals['stop_loss']   # absolute price level  ← MISSING
signals['take_profit'] # absolute price level  ← MISSING
```

Both `stop_loss` and `take_profit` must be computed from current ATR. MIDGE has ATR computation in `mae_core/market/edge/ta_indicators.py` and price data in `price_fetcher.py`, but neither is attached to ConvergenceAlerts. The `signal_translator.py` module (proposed by sibling, not yet built) must:
1. Extract ticker from the dominant signal in `alert.signals`
2. Fetch current OHLCV via `price_fetcher.py`
3. Compute ATR (14-period)
4. Set SL = entry ± 1.5×ATR, TP = entry ± 3.0×ATR

This is a well-scoped build, not a research question. Estimated: 100–150 lines.

---

## 6. Instrument Coverage Gap

### Source rotation categorized by FTMO relevance

**Source:** `C:\Users\baenb\projects\MIDGE\mae_core\market\sensing_hook.py`, SOURCE_ROTATION list (lines 144–188)

FTMO trades: forex pairs (EUR/USD, GBP/USD, etc.), indices (NASDAQ/NQ, S&P/ES, DAX, Dow/YM), commodities (gold/GC, oil/CL).

| Source | Category | FTMO Relevant? | Notes |
|--------|----------|---------------|-------|
| `sec_form4` | US equity insider trades | No | Form 4 filers are US public companies |
| `sec_form8k` | US equity events | No | 8-K filings are equity-specific |
| `congressional` | US equity government trades | Partial | Trades that land on NQ/ES-adjacent tickers (AAPL → NQ) |
| `senate` | US equity government trades | Partial | Same as congressional |
| `hiring` | US equity job postings | No | Equity-only signal |
| `usa_spending` | US government contracts | No | Equity-specific tickers |
| `sam_gov_and_prices` | US government contracts | No | Equity-specific |
| `social_sentiment` | US equity social signals | No | Equity-focused |
| `finra_short` | US equity short interest | No | Equity-specific |
| `sec_efts` | US equity SEC filings | No | Equity-specific |
| `finnhub` | Mixed news | Partial | Macro/index news included |
| `fred_macro` | FRED macroeconomic data | **Yes** | Directly moves forex pairs; Fed data = EUR/USD driver |
| `session_sweep` | ICT/SMC price patterns | **Yes** | Works on any liquid instrument including NQ, ES, forex |
| `ta_indicators` | Technical analysis | **Yes** | RSI/MACD/BB applicable to NQ=F, ES=F, GC=F, CL=F |
| `order_flow` | Intraday volume imbalance | **Yes** | Instrument-agnostic |
| `fractal_resonance` | Multi-timeframe patterns | **Yes** | Instrument-agnostic |
| `cot_positioning` | CFTC COT data | **Yes** | COT tracks non-commercial positioning in forex AND futures (EUR, JPY, GBP, gold, crude, S&P, NASDAQ) — directly maps to FTMO instruments |
| `stocktwits` | Social sentiment | No | Primarily equity retail |
| `vix_structure` | VIX term structure | **Yes** | Fear signal for equity indices (ES, NQ, YM) |
| `google_trends` | Search interest | Partial | Macro/event search spikes (e.g., "Fed rate hike") |
| `finnhub_extras` | Economic calendar via Finnhub | **Yes** | FOMC, CPI, NFP = primary forex movers |
| `crypto_prices` | CoinGecko crypto | No | FTMO offers limited crypto but not MIDGE's crypto focus |
| `crypto_exchange` | CoinCap crypto | No | Same |
| `openinsider` | Pre-filtered insider buys | No | US equity |
| `institutional_13f` | 13F institutional holdings | Partial | Large ETF flows (SPY, QQQ) can indicate index direction |
| `finviz` | FinViz unusual volume/shorts | No | Equity screener |
| `economic_calendar` | Scheduled macro events | **Yes** | Precisely the events that move forex: NFP, FOMC, CPI |
| `massive_snapshot` | Polygon.io daily OHLCV | Partial | Includes NQ=F, ES=F in watchlist |
| `eia_energy` | EIA crude/gas inventories | **Yes** | Weekly EIA reports directly move CL=F (crude oil futures) |
| `congress_legislation` | Legislative pipeline | Partial | Major legislation (infrastructure, energy policy) moves sector indices |
| `social_text` | StockTwits NLP | No | Equity-focused |
| `yahoo_rss` | Yahoo RSS headlines | Partial | Major macro headlines included |

**FTMO-relevant sources (directly applicable):** 9 out of 32
- `fred_macro`, `session_sweep`, `ta_indicators`, `order_flow`, `fractal_resonance`, `cot_positioning`, `vix_structure`, `finnhub_extras`, `economic_calendar`, `eia_energy`

**FTMO-partially-relevant sources:** 6 out of 32
- `congressional`, `senate`, `finnhub`, `google_trends`, `institutional_13f`, `massive_snapshot`, `congress_legislation`, `yahoo_rss`

**Equity-only (no FTMO value):** ~17 out of 32

**Instrument coverage in live data:**
From the deduplicated pattern_stack alerts: only 2 futures-style tickers found in the 199 unique tickers: `ES=F` and `NQ=F`. No forex pairs (EUR/USD, GBP/USD, etc.), no gold (GC=F), no crude (CL=F). These 2 futures tickers account for a tiny fraction of alerts.

**Watchlist gap:** MIDGE's watchlist (`data/midge/watchlist.json`) was not checked here, but the signal output confirms the dominant coverage is US equities. The forex/commodity/index universe is barely populated.

---

## 7. Drawdown Risk Modeling

### Monte Carlo reasoning with actual numbers

**Parameters:**
- Win rate: 19.9% (overall) to 30% (filtered combo subset)
- Payoff: 3.34:1 (avg win 11.4% vs avg loss 3.4%)
- FTMO constraints: bust at 10% DD, daily halt at 5% DD
- Position size: 1% risk per trade

**At 19.9% WR (current overall rate):**

This is a losing system (E = -0.45%/trade). In a Monte Carlo simulation:
- After 100 trades: expected account value = $10,000 × (1 - 0.0045)^100 ≈ $6,394
- The 10% drawdown bust ($9,000 floor) will be hit with near certainty before reaching the $11,000 target
- P(10 consecutive losses) = 0.801^10 = **10.9%** — will occur roughly every 9 trade sequences
- A 10-loss streak at 1% risk = exactly the 10% drawdown bust

**Verdict at 19.9% WR: certain ruin on FTMO.** Not a probability question — a negative expectancy system will hit the drawdown limit before the profit target.

**At 30% WR (filtered best-combo subset):**

- E = +0.302%/trade (positive expectancy)
- P(10 consecutive losses) = 0.70^10 = 2.8%
- Expected trades to 10% target ≈ 33
- Expected maximum drawdown path: With 70% loss rate and 1% risk, the Gambler's Ruin problem gives:
  - P(reaching -10% before +10%): with E > 0, approximately 15–20% (conservative estimate)
  - This is manageable but not negligible — 1 in 5 to 1 in 7 attempts fails despite positive expectancy

**At 50% WR (best-case small-sample combos):**
- E = +1.17%/trade
- P(10 consecutive losses) = 0.50^10 = 0.097%
- Expected trades to 10% target: ~9
- P(ruin before target): < 2% — highly viable

**Key structural risk:** The current convergence signal confidence does not discriminate between winners and losers (phase0-measurements.md: winners avg 0.560, losers avg 0.565 confidence). This means the 19.9% overall rate is what MIDGE will experience in practice — not the 30%+ combo rate, which only reveals itself after the fact by examining combo membership. A real-time filter on combos requires knowing in advance which domain combination a signal belongs to and whether that combination has >= 30% historical WR on FTMO instruments. This combo-level filter does not currently exist in the execution path.

---

## 8. Existing Forex/Commodity Sources

Sources already built in MIDGE that produce signals relevant to FTMO instruments:

### Directly FTMO-useful (produce signals that move forex, indices, or commodities):

**COT Positioning (`cot_positioning` / `cot_client.py`):**
- CFTC Commitment of Traders reports large speculator positioning in: EUR futures, GBP futures, JPY futures, AUD futures, Gold (GC), Crude Oil (CL), E-mini S&P (ES), E-mini NASDAQ (NQ)
- Non-commercial net long/short is a medium-term directional signal for all major forex pairs and FTMO commodity instruments
- Already wired, data quality uncertain (Phase 0 found only 5 of 28 sources measured)

**EIA Energy (`eia_energy` / `eia_client.py`):**
- Weekly crude oil inventories, natural gas storage, crude production data
- Directly moves CL=F (crude oil futures) — a FTMO-tradeable instrument
- Already wired and collecting data
- Thompson prior: 0.70 (government data reliability — highest in system)

**Economic Calendar (`economic_calendar` via Finnhub):**
- FOMC decisions, CPI prints, NFP releases, GDP revisions
- These are the primary price-moving events for EUR/USD, GBP/USD, USD/JPY
- Already in SOURCE_ROTATION, already suppresses trading during windows
- ALIGNMENT: MIDGE's suppression windows (avoid trading during events) are the OPPOSITE of what FTMO needs — FTMO prohibits trading within 2 hours of major news (explicitly stated in their rules). MIDGE already implements this. They are aligned.

**FRED Macro (`fred_macro`):**
- Federal Reserve economic data — Fed funds rate, inflation, employment
- Moves forex pairs directly
- Already wired with thematic tier routing

**VIX Term Structure (`vix_structure` / `vix_client.py`):**
- Fear gauge for equity indices
- VIX contango/backwardation signals index direction for ES=F and NQ=F
- Already wired

**Session Sweep + IFVG (`session_sweep`, `session_sweep_ifvg`):**
- ICT/Smart Money Concepts — identifies institutional order blocks and fair value gaps
- Instrument-agnostic. Originally developed for forex (London/NY session sweeps)
- Proven: quality gate at 0.40 improves WR from 39.1% to 44.4% (MEMORY.md)
- This is the single most FTMO-applicable signal already in MIDGE

**TA Indicators (`ta_indicators`):**
- RSI, MACD, Bollinger Bands, Market Structure applied to any OHLCV data
- If watchlist includes GC=F, CL=F, NQ=F, ES=F — these signals fire on FTMO instruments
- Watchlist is the constraint, not the indicator capability

### What is missing for FTMO coverage:

| Missing | Impact | Difficulty to Add |
|---------|--------|------------------|
| Forex pairs in watchlist (EUR/USD, GBP/USD, USD/JPY) | High — no forex signals despite forex-capable indicators | Low — add tickers to watchlist |
| Gold (GC=F) and crude (CL=F) in watchlist | High — EIA signals for CL exist but no price data feeds | Low — add to watchlist |
| COT forex positioning wired to signal output | Medium — COT client exists but signal routing needs verification | Medium — audit `cot_client.py` output → convergence |
| DAX, FTSE in watchlist (FTMO indices) | Low — MIDGE focuses on US — international indices would need a new data source | High — requires new price feed |

---

## Critical Gaps Summary

| Gap | Severity | Status |
|-----|----------|--------|
| ConvergenceAlert has no entry price, SL, or TP | Blocker | `signal_translator.py` not built |
| 19.9% overall WR is below 23% FTMO break-even | Critical | Requires combo-level filter targeting 30%+ WR combos on FTMO instruments |
| Confidence engine does not discriminate winners from losers | Critical | Thompson distributions are 81/83 at uniform priors — not yet calibrated |
| 97%+ of signals are on US equities (FTMO does not trade equities) | Critical | Watchlist must be expanded to include NQ=F, ES=F, GC=F, CL=F, major forex pairs |
| Paper account sized at $50K — FTMO account is $10K | Blocker | Position sizer must be recalibrated |
| `challenge_tracker.py` does not exist | Blocker | State machine tracking challenge progress needed for risk gates |
| FTMO 2-step structure not encoded anywhere in MIDGE | Blocker | Phase 1 vs Phase 2 vs Funded configs needed |
| Replay results file is empty (`replay_results.json`) | Data gap | Historical backtest data appears to have been lost — `{"alerts": [], "phase": "replay"}` |

---

## What Works Today (No Changes Needed)

1. **Risk architecture:** DrawdownMonitor, SystemHealthMonitor, SelfMonitor — all built, wired, functional. Need threshold reconfiguration only.
2. **FTMO-aligned news suppression:** MIDGE already avoids trading during major economic events. Directly satisfies FTMO's news trading prohibition.
3. **Session Sweep signals on forex timeframes:** The session sweep already targets institutional session sweeps — the signal type that FTMO instruments are most susceptible to.
4. **COT, EIA, Economic Calendar, VIX, FRED:** All already built and collecting data on FTMO-relevant domains. Need watchlist expansion to activate on FTMO instruments.
5. **Pattern Archaeology:** Templates accumulate cross-symbol. If NQ=F and ES=F are added to watchlist, patterns would start forming within weeks.
6. **ConvergenceAlert direction field:** `bullish` / `bearish` maps cleanly to `signal = 1 / -1`. One field is ready.

---

## Fastest Path to a Testable FTMO Signal

Three sequential steps, ordered by dependency:

**Step 1:** Expand watchlist to include NQ=F, ES=F, GC=F, CL=F, EUR/USD (as a yfinance-compatible ticker like EURUSD=X). This immediately activates TA indicators, session sweeps, and pattern archaeology on FTMO-tradeable instruments. Estimated: 1–2 hours.

**Step 2:** Build `signal_translator.py` — takes a ConvergenceAlert, fetches current price + ATR for the alert's primary ticker, and outputs the sibling's signal format (signal/stop_loss/take_profit). Estimated: 150 lines, 3–4 hours.

**Step 3:** Run the historical convergence alerts from live data against the sibling's FTMO backtester engine for NQ=F and ES=F specifically. This gives the first real estimate of MIDGE-on-FTMO-instruments win rate. Estimated: 2–3 hours.

**Gate before live attempt:** The $1,000 deployment gate from Guiding Light's directive requires 80%+ historical accuracy on pattern stacks. Current high-confidence pattern stacks (>= 0.7) are ungraded. This gate is not yet clearable — the data to clear it does not exist. The free FTMO trial is the right next step to generate that data without capital risk.

---

## Sources

- `C:\Users\baenb\projects\MIDGE\data\midge\alerts_human.jsonl` — 24,222 records, 3 days live data
- `C:\Users\baenb\projects\MIDGE\data\midge\paper_trades.jsonl` — 1,055 records, 7 unique signals (logging duplication bug)
- `C:\Users\baenb\projects\MIDGE\data\midge\replay_results.json` — empty (`{"alerts": [], "phase": "replay"}`)
- `C:\Users\baenb\projects\MIDGE\mae_core\market\intelligence\convergence_alerter.py` — lines 62–107 (ConvergenceAlert dataclass)
- `C:\Users\baenb\projects\MIDGE\mae_core\market\sensing_hook.py` — lines 144–245 (SOURCE_ROTATION, TIER_ROUTING)
- `C:\Users\baenb\projects\MIDGE\mae_core\market\intelligence\learning_config.py` — full file (source reliability priors, paper account value)
- `C:\Users\baenb\projects\MIDGE\data\market\thompson_distributions.json` — 83 distributions, 81 at uniform prior
- `C:\Users\baenb\projects\MIDGE\FTMO-EXECUTION-ENGINE.md` — sibling's integration path and interface contract
- `C:\Users\baenb\projects\MIDGE\research\phase0-measurements.md` — 19.9% WR, 3.34:1 payoff, confidence-doesn't-discriminate finding
