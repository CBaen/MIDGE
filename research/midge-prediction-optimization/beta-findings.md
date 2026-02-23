# MIDGE Prediction Optimization — Beta Findings
## Witness: Market Microstructure, Timing, and Execution

**Analytical Lens:** When signals matter, how quickly they decay, what optimal holding periods are, how to size positions, and how to turn signals into profitable trades.

**Date:** 2026-02-22

---

## Executive Summary

MIDGE has strong signal detection machinery but is architecturally blind to the dimension that determines whether that machinery produces profit or noise: time. Every hardcoded decay rate is wrong. The 72-hour convergence window is wrong for some signals and too tight for others. The system has no position sizing, no transaction cost awareness, and no mechanism for distinguishing between leading indicators (information not yet priced in) and lagging confirmations (information the market already knew). Most critically, the system conflates signals that require action within hours with signals that are valid for months, treating them identically in a single convergence window. This analysis maps each signal to its actual information half-life and provides specific, empirically grounded numbers throughout.

---

## 1. Signal Timing Analysis — Information Half-Life by Source

This section maps each of MIDGE's 8 data sources against empirical research on how quickly markets price in that information.

### 1.1 SEC Form 4 Insider Trades — decay_rate currently 0.05/day

**Current state:** `InsiderTrade.decay_rate = 0.05`, implying a half-life of ~14 days (ln(2)/0.05). The `MarketSignal.outcome_window_days` defaults to 14.

**Empirical reality:** Lakonishok and Lee (2001) remain the canonical study. Key findings:
- Firms with extensive insider purchases beat firms with extensive insider sales by 7.8% over the following 12 months (4.8% after controlling for size and book-to-market).
- **One quarter of these abnormal returns accrued within the first 5 trading days** after the transaction was disclosed.
- **One half of the total 12-month alpha accrued within the first month.**

This means the information is fast-decaying early and slow-decaying later. It is not well-modeled by a single constant decay rate.

**What 0.05/day actually models:** A half-life of 14 days. If 50% of the alpha is captured in 20 trading days (~28 calendar days), the implied decay should be faster: ~0.025/day for the sustained component of the trade. But the first-5-days component decays at roughly 0.15-0.20/day.

**Correct model:** Two-component decay.
- Fast component (market reaction to disclosure): decays at ~0.15/day, responsible for approximately 25% of total signal.
- Slow component (fundamental information): decays at ~0.02/day, responsible for approximately 75% of total signal.
- A usable single-rate approximation: **0.035/day** for the full position, implying a 20-day half-life.

**Optimal holding period:** 20 to 60 trading days. Entries within 2 trading days of Form 4 disclosure capture the fast component. Holding beyond 60 days adds diminishing return per unit of time held.

**MIDGE's current 14-day outcome window** is evaluating trades before the main alpha period. Set `outcome_window_days = 45` for insider trades.

**Latency window:** SEC EDGAR submissions API updates with sub-second processing delay (SEC developer documentation). MIDGE's current polling is manual. Form 4 filings should be detected within 1-4 hours of filing via the EDGAR RSS feed. The market reaction begins within minutes of publication by financial data aggregators (Bloomberg, Reuters, etc.), meaning if MIDGE detects in the same day but not the same hour, it is likely entering after the fast first-day move has already occurred.

**Timeline mapping:** 0-2 hours post-filing: fastest 15-25% of alpha available. 2-5 trading days: next 10-15%. Weeks 2-8: remaining sustained alpha.

---

### 1.2 Insider Buying Clusters (ClusterSignal) — decay_rate currently 0.05/day

**Current state:** `ClusterSignal.decay_rate = 0.05`, same as individual insider trades.

**Empirical reality:** Cluster buys are materially stronger signals than individual trades. Research (Alldredge 2019, Kang, Kim, Wang) shows:
- Clustered insider purchases with peer trades within 2 days generate 2.1% abnormal returns over the next month, vs 1.2% for solitary purchases — approximately 75% higher alpha.
- Over 21 trading days, cluster buys returned 3.8% vs 2.0% for non-cluster purchases.
- Over 90 trading days, the gap widens to a 2.5% cumulative difference.

**Correct decay rate:** Cluster buys should have a **slower** decay than individual trades — not faster. The information is more persistent because multiple independent confirmation points reduce the probability of noise. Suggested: **0.025/day**, implying a 28-day half-life.

**Outcome window:** Set to 60 days, not 14.

**Minimum threshold for cluster validity:** The current 30-day cluster window in `cluster_detector.py` is appropriate. The requirement for minimum 3 insiders is consistent with the research showing the most significant signal amplification starts at 3+ independent buyers.

---

### 1.3 Congressional Trades — decay_rate currently 0.03/day

**Current state:** `CongressionalTrade.decay_rate = 0.03`, implying a half-life of ~23 days. Confidence baseline: 0.65.

**Empirical reality:** The STOCK Act requires disclosure within 45 days of the transaction. This means by the time MIDGE sees a congressional trade, the information asymmetry window between the trade and the public knowledge is already 1-45 days old. However:
- Research (PMC 2022 study) found senators earn approximately **4.9% market-adjusted returns over 3 months** based on their trades.
- The information asymmetry around senator trades is significantly greater than around earnings announcement days, implying this is genuinely informed trading.
- Critically: the asymmetry persists for an extended period because legislative/regulatory information is slow to be fully priced in.

**The disclosure lag problem:** When MIDGE receives a congressional trade via House Stock Watcher, it is seeing disclosure, not the original transaction. The actual event happened up to 45 days ago. The signal strength should be attenuated by how stale it is:
- If disclosed within 5 days of transaction: full signal value (~0.65 confidence)
- If disclosed 30-45 days after transaction: signal value degraded by ~50% (~0.35 confidence)

MIDGE's current code correctly distinguishes `timestamp` (transaction_date) from `received_at` (disclosure_date) in the signal adapter, but the convergence alerter uses `timestamp` when pruning signals. **The effective decay should be applied from received_at, not from timestamp.** The original trade is old news. What is actionable is the disclosure event.

**Correct decay rate from disclosure date:** 0.05/day — faster decay than the current 0.03, because by the time of disclosure, much of the move is already behind you. The initial "news" of the disclosure (market learning about the trade) decays very quickly (2-3 days). The sustained informational component is valid for 30-60 more days.

**Optimal holding period:** 15-45 days from disclosure date (not transaction date). High-value political intelligence (committee chairs trading their own oversight sectors) is worth holding longer.

**Transaction costs kill sub-$50,000 trades:** The scan report shows dozens of congressional trades in the $8,000 range (str=0.02). At typical transaction costs of 0.5-2% round-trip plus slippage on entry, a $8,000 congressional trade signal is economically worthless. The minimum actionable congressional signal is approximately $50,000 in trade size, and ideally $250,000+.

**What the current scan shows:** The 2026-02-22 scan shows 100 congressional signals. Of these, the vast majority are $8,000-$32,500 trades (strength 0.02-0.07). Only 1 trade (Kevin Hern buying $375,000 of RY) has meaningful signal weight. The system is generating noise, not signal, from congressional data.

---

### 1.4 Government Contract Awards (USASpending) — decay_rate 0.03/day (via GovernmentContract)

**Current state:** `contract_award` signals use the `GovernmentContract` model. The strength formula is `min(1.0, contract.award_amount / 100_000_000)` — so a $50M contract gets strength 0.50.

**Timing characteristics:** Contract awards are public events. When USASpending publishes an award, the information is already public. The question is whether MIDGE detects it quickly enough to trade before the market prices it in.

**DoD contract announcements:** The Department of Defense announces contracts valued at $7.5M+ at 5 PM Eastern each business day. This is a scheduled, predictable publication event. The information is fully public the moment it is posted.

**Actual alpha window:** Defense contractors are large-cap, heavily followed stocks. An announcement of a $100M contract on a $50B market-cap company (LMT at $658: market cap ~$160B) is a 0.06% revenue event. The alpha from a single publicly announced award is marginal for large caps.

**Where the real alpha is:** Pre-announcement prediction (the ContractPredictor). If MIDGE can identify probable winners 4-12 weeks before announcement, the holding period is the time from prediction to announcement, then several weeks after for the sustained post-announcement drift.

**Decay rate for post-announcement award signal:** 0.08/day — faster than current 0.03, because the market prices in the announcement within 1-3 trading days for large caps.

**Decay rate for pre-announcement prediction signal:** 0.02/day — very slow, because the hypothesis takes weeks to resolve. The `ContractPrediction.decay_rate = 0.03` is close but slightly too fast.

---

### 1.5 SAM.gov Contract Opportunities — hardcoded decay_rate 0.04/day

**Current state:** `from_contract_opportunity()` hardcodes `decay_rate=0.04`, strength fixed at 0.30, confidence 0.40.

**Timing characteristics:** SAM.gov opportunity postings are the announcement that a contract is being competed. This is early-stage intelligence — the award decision may be 3-18 months away. The alpha source is: (1) knowing who the likely winners are before the contract closes; (2) accumulating position over the competition period.

**Information half-life:** A SAM.gov opportunity posting is valid as an investment thesis for the entire duration of the competition (months to over a year). Decay rate should be approximately **0.008/day** (half-life of ~87 days), not 0.04.

**Recommended scan frequency:** Weekly is sufficient for SAM.gov opportunities. The data does not change intra-day in a way that requires more frequent polling.

---

### 1.6 Job Tracker / Hiring Signals — decay_rate 0.07/day

**Current state:** `HiringSignal.decay_rate = 0.07`, implying a ~10-day half-life. Confidence baseline 0.50.

**Timing characteristics:** A hiring blitz is a leading indicator. The key insight (as noted in the codebase's docstring) is that companies begin hiring for work they expect to win before the contract award. The alpha window is therefore:
- From hiring spike detection to contract announcement: weeks to months (speculative phase)
- Post-announcement: rapid re-rating, then decay

**Decay rate:** The current 0.07/day is too fast for the pre-announcement phase and about right for the post-announcement phase. For pre-announcement use (where hiring is predicting a future contract win), the decay should be **0.01-0.02/day**.

**What the scan shows:** RTX showing str=0.84 conf=0.85 for hiring is the strongest signal in the most recent scan. This is paired with RTX bearish Form 4 data (insiders selling). The divergence between hiring bullish (contract pipeline expectation) and insider selling (compensation plan execution) is exactly the kind of nuance that requires timeframe-aware analysis — these signals are at different phases of the same corporate cycle.

**Lead time empirically:** In government contracting, major program hires typically begin 60-120 days before a contract announcement. The informational value of the signal is valid for that entire lead-time window.

---

### 1.7 SEC Form 8-K Material Events — decay_rate 0.03/day (from Form8KEvent)

**Current state:** `Form8KEvent.decay_rate = 0.03`, confidence 0.50-0.70.

**Timing characteristics:** 8-K filings must be made within 4 business days of a material event. The event itself already occurred up to 4 days before filing. Market reaction to 8-Ks is extremely fast — within minutes to hours of filing.

**Critical insight:** The current scan shows all 8-K signals as "neutral" (str=0.70, conf=0.70 across the board). This is because the item code direction mapping is heavily "neutral" — items like 5.02 (Officer/Director Changes), 2.02 (Results of Operations), 7.01 (Reg FD) are all mapped to neutral. This discards enormous informational content. An 8-K about officer departure at a critical time is very different from a routine Reg FD disclosure.

**For truly material events (1.01 agreements, 1.03 bankruptcy, 2.06 impairment):** These are binary events. The market reaction happens within the trading day of filing. Alpha is captured in hours, not days. These signals should have decay_rate **0.30/day or higher** and `outcome_window_days = 5`.

**For informational events (7.01 Reg FD, 8.01 Other):** Neutral assignments are appropriate. These do not generate actionable signals by themselves.

---

### 1.8 Filing Time Analyzer — decay_rate 0.10/day

**Current state:** `FilingTimeSignal.decay_rate = 0.10`, confidence modifiers from -0.15 to +0.05.

**Timing characteristics:** Filing time behavioral signals are modifiers to other signals, not standalone signals. The Friday afternoon trash-dump pattern has been validated in academic literature (empirical studies on Friday earnings announcements showing negative drift). The decay rate of 0.10/day is appropriate — the behavioral context of when something was filed becomes irrelevant after a week.

**Assessment:** This is correctly calibrated. The patterns are:
- Friday 3-5 PM (hiding): -15% confidence modifier. Correct.
- After hours (avoiding): -8% modifier. Correct.
- Pre-market (urgent): +5% modifier. Appropriate.

The one gap: the analyzer is not being fed into MIDGE's signal strength calculation in any live path. It produces `FilingTimeSignal` objects but these are not converted to `MarketSignal` objects and do not reach the ConvergenceAlerter. The data flows to Qdrant query only, not to the active scan pipeline in `midge_scan.py`. This means a perfectly timed Friday dump filing is not getting its confidence discounted in the convergence alert.

---

## 2. Scan Frequency Optimization

**Current state:** MIDGE runs once manually via `python midge_scan.py`. The `midge_scan.py` uses `convergence_window_hours=168` (7 days), while the bootstrapped ConvergenceAlerter in `main.py` uses 72 hours.

**The fundamental problem:** Different sources have radically different update cadences. Lumping them into one scan at one frequency wastes compute on slow sources and loses alpha on fast sources.

### Recommended Scan Schedule

| Source | Recommended Frequency | Reason |
|--------|----------------------|--------|
| SEC Form 4 (EDGAR RSS) | Every 1-4 hours during market hours | EDGAR processes filings near real-time; early detection captures the fast-decay component |
| SEC Form 8-K material events | Every 1-4 hours during market hours | Same as Form 4; some 8-Ks are time-critical |
| Congressional trades | Once daily (after market close) | Disclosures batch-publish; intra-day polling adds no value |
| Government contract awards (USASpending) | Every 6-12 hours on business days | DoD announces at 5 PM; once-daily post-market check is sufficient |
| SAM.gov opportunities | Once weekly | Competition periods are measured in months; daily polling is waste |
| Job tracker / hiring signals | Once daily | Job postings accumulate over days; intra-day volatility is noise |
| Price data | Continuous (60-second cache already implemented) | Already handled correctly |

**Implementation approach:** Rather than one monolithic scan, MIDGE needs a tiered scheduler:
- **Tier 1 (every 2 hours):** SEC EDGAR Form 4 and 8-K RSS polling
- **Tier 2 (daily, after market close):** Congressional trades, contract awards, hiring signals
- **Tier 3 (weekly):** SAM.gov opportunities, SAM.gov open solicitations

The `ConvergenceAlerter` should maintain separate time windows per domain: 48 hours for SEC filings, 72 hours for congressional, 7 days for contracts and hiring.

---

## 3. Decay Rate Calibration Summary

All current decay rates vs. empirically justified values:

| Signal Source | Current decay_rate | Recommended decay_rate | Implied Half-Life | Basis |
|--------------|-------------------|----------------------|-------------------|-------|
| InsiderTrade | 0.05 | 0.035 | 20 days | Lakonishok/Lee: 50% alpha within 20 trading days |
| ClusterSignal | 0.05 | 0.025 | 28 days | Cluster buys persist longer; 90-day window shows increasing gap |
| CongressionalTrade | 0.03 | 0.05 (from disclosure) | 14 days | Trade is stale by disclosure; 45-day lag erodes value |
| Form8KEvent (material) | 0.03 | 0.25 | 3 days | Market prices binary events within trading day |
| Form8KEvent (neutral/informational) | 0.03 | 0.05 | 14 days | Moderate persistence |
| FilingTimeSignal | 0.10 | 0.10 | 7 days | Correctly calibrated |
| ContractPrediction | 0.03 | 0.018 | 39 days | Pre-announcement thesis valid for weeks to months |
| GovernmentContract (award) | 0.03 | 0.07 | 10 days | Post-announcement large-cap drift resolves quickly |
| HiringSignal (pre-announcement) | 0.07 | 0.015 | 46 days | Hiring leads contract by 60-120 days; slow decay |
| HiringSignal (standalone/no contract context) | 0.07 | 0.07 | 10 days | Without contract thesis, hiring signal is weaker |
| SAM.gov Opportunity | 0.04 | 0.008 | 87 days | Competition periods last months |
| CorrelationSignal (politician) | 0.03 (hardcoded in signal.py) | 0.04 | 17 days | Combination of trade staleness and political info persistence |

---

## 4. Position Sizing and Portfolio Construction

**Current state:** MIDGE generates convergence alerts with strength (0-1) and confidence (0-1) fields, but has no concept of how much capital to allocate to any idea. The `TradeSignal` dataclass has no position size, no dollar allocation, no risk limit.

### 4.1 Why No Position Sizing Is a Critical Gap

The current scan report shows a BULLISH convergence alert at strength=0.92, confidence=0.82 for insider + institutional domains. Without position sizing logic, this high-confidence alert generates the same response as a strength=0.60, confidence=0.55 alert: a human has to decide what to do with it. The system cannot be automated, and it cannot apply consistent risk management.

### 4.2 The Kelly Criterion Framework for MIDGE

The Kelly Criterion determines the fraction f* of capital to risk on a single trade:

```
f* = (p * b - q) / b
```

Where:
- p = probability of win (confidence from convergence alert)
- q = 1 - p
- b = ratio of profit to loss (expected_gain / max_loss)

**Practical adjustments:** Full Kelly is mathematically optimal but emotionally brutal and practically dangerous due to model error. The standard institutional practice is **fractional Kelly at 25-50%** of the full Kelly fraction.

**Example mapping MIDGE outputs to Kelly:**
- A convergence alert at confidence=0.82, strength=0.92, 3 domains converging
- Assume estimated gain (from signal research) = 4.8% over 45 days
- Assume max acceptable loss = 8% (stop-loss)
- b = 4.8 / 8.0 = 0.60
- Full Kelly: f* = (0.82 * 0.60 - 0.18) / 0.60 = (0.492 - 0.18) / 0.60 = 0.52
- At 25% Kelly (conservative): allocate 13% of portfolio
- At 50% Kelly (moderate): allocate 26% of portfolio

**Recommended implementation:** Add a `position_size_pct` field to `TradeSignal`, calculated as:
```python
kelly_fraction = (confidence * expected_return_ratio - (1 - confidence)) / expected_return_ratio
position_size_pct = min(kelly_fraction * kelly_divisor, max_position_pct)
```

Where `kelly_divisor = 4` (25% Kelly) and `max_position_pct = 0.15` (15% hard cap per position).

### 4.3 Cross-Signal Portfolio Construction

MIDGE's domain-category architecture (behavioral, market, social, information, financial, institutional) is actually designed for risk diversification — signals from different categories are less correlated. This maps naturally to risk parity principles:

- Allocate total portfolio risk equally across category-diversified convergence signals
- Scale down when same-category signals dominate (e.g., 5 insider signals all pointing the same direction may be correlated noise, not 5 independent data points)
- The `cross_domain_count` field in `ConvergenceAlert` is the correct diversity measure — higher cross_domain_count supports higher position sizing

**Recommended rule:** Position size scales with `cross_domain_count`:
- 2 domains: 0.5x base size
- 3 domains: 1.0x base size
- 4+ domains: 1.5x base size (with hard cap)

---

## 5. Execution Strategy

**Current state:** The `urgency` field in `ConvergenceAlert` classifies alerts as "immediate," "hours," or "days" based on velocity (threshold 0.1 for immediate, 0.05 for hours). No execution guidance is attached.

### 5.1 Urgency Classification Analysis

The velocity thresholds are arbitrary. Looking at the code:

```python
if avg_velocity > 0.1:
    urgency = "immediate"
elif avg_velocity > 0.05:
    urgency = "hours"
else:
    urgency = "days"
```

The velocity values here are signal velocity (rate of change of signal strength per day), not market velocity. An avg_velocity of 0.1 on a scale of 0-1 means the signal is strengthening at 10% of its range per day — a fairly slow movement. The thresholds may incorrectly classify events.

**More meaningful urgency classification** should incorporate:
1. Signal source type (8-K = always immediate; congressional disclosure = days; SAM.gov = days-to-weeks)
2. Whether the underlying event is time-sensitive (earnings announcement in 2 days = immediate)
3. Velocity of convergence (how quickly domains are aligning)

### 5.2 Execution Method by Urgency

| Urgency Level | Execution Method | Rationale |
|---------------|-----------------|-----------|
| Immediate (8-K material event, same-day Form 4) | Market order or aggressive limit | Alpha window measured in hours; slippage is worth paying |
| Hours (insider cluster just filed, strong velocity) | Limit order within bid-ask spread | 4-8 hour execution window; some slippage tolerance |
| Days (congressional trade, contract award) | VWAP or TWAP over 1-2 days | No urgency; minimize market impact; multi-day accumulation |
| Weeks (SAM.gov opportunity, hiring blitz early detection) | Staged entry over 1-2 weeks | No urgency at all; use limit orders at various price levels |

**Current problem observed in scan data:** MIDGE's most recent convergence alert was classified "days urgency" for a bullish insider + institutional (hiring) convergence. This is correct — RTX hiring blitz + some insider buying is a medium-term thesis, not an intraday trade.

### 5.3 TWAP vs. VWAP for MIDGE's Signal Types

- **VWAP (Volume-Weighted Average Price):** Optimal when you want to trade proportionally to market activity throughout the day. Best for larger position sizes where minimizing market impact matters.
- **TWAP (Time-Weighted Average Price):** Optimal when the goal is simply to distribute execution over time without prediction of volume patterns. Simpler and more appropriate for MIDGE's use case.

**Recommendation:** TWAP over 1-2 days for "days" urgency alerts. Market order entry for "immediate" alerts. This is sufficient for the retail/semi-institutional scale MIDGE operates at. Full VWAP implementation requires access to intraday volume profiles and adds significant complexity for marginal improvement at this scale.

---

## 6. Multi-Timeframe Architecture

**Current state:** A single convergence window. The bootstrapped `ConvergenceAlerter` uses 72 hours. `midge_scan.py` uses 168 hours. Both are single-window.

**The fundamental flaw:** The system conflates:
- Form 4 insider filing (relevant for 20-60 days, time-sensitive within first 24-48 hours)
- SAM.gov opportunity (relevant for months, not time-sensitive at all)
- Hiring blitz (relevant for the duration of the competition period, measured in months)

All three feeding the same 72-hour window means hiring signals from 2 weeks ago and insider filings from yesterday are competing as equally fresh signals.

### Recommended Multi-Timeframe Architecture

Three independent convergence analyzers running simultaneously:

**Tier 1: Tactical (48-72 hour window)**
- Feeds: SEC Form 4, SEC Form 8-K material events
- Purpose: Captures fast-decay alpha from disclosure events
- Minimum confidence for alert: 0.70
- Alert frequency: Can fire multiple times per week

**Tier 2: Strategic (14-21 day window)**
- Feeds: Congressional trades (from disclosure date), contract awards, insider clusters, politician correlation signals
- Purpose: Medium-term event-driven positions
- Minimum confidence for alert: 0.65
- Alert frequency: 1-2 per week maximum

**Tier 3: Thematic (60-90 day window)**
- Feeds: SAM.gov opportunities, hiring blitzes, contract predictions
- Purpose: Long-duration themes (defense contractor pipeline, sector rotation)
- Minimum confidence for alert: 0.60
- Alert frequency: Monthly

Each tier generates `ConvergenceAlert` objects with a `tier` field, and position sizing scales with tier (Tier 3 = largest position, longest hold, lowest urgency).

---

## 7. Transaction Cost Awareness

**Current state:** MIDGE has no concept of transaction costs. The `min_strength = 0.6` threshold in `ConvergenceAlerter` and `min_price_move_pct = 2.0` in `OutcomeTracker` are the only economic filters, and they are not linked.

### 7.1 The Break-Even Signal Strength Problem

For a signal to be worth acting on after transaction costs, the expected return must exceed the total trading cost.

Estimated round-trip trading costs:
- Large cap (AAPL, MSFT, NVDA): commission ~$0 (most retail brokers), bid-ask spread 0.01-0.05%, slippage 0.0-0.1%. Total: **0.05-0.15% round-trip**.
- Mid cap (RTX, NOC, LMT): spread 0.02-0.10%, slippage 0.05-0.20%. Total: **0.10-0.30% round-trip**.
- Small cap: spread 0.1-0.5%, slippage 0.1-1.0%. Total: **0.25-1.5% round-trip**.

For a congressional trade signal with expected alpha of 4.9% over 90 days:
- Required signal strength to break even: alpha must exceed 0.15% (large cap) to 0.30% (mid cap)
- A $8,000 congressional trade generating an alert with str=0.02 is not worth acting on, period

**The scan report problem illustrated:**
The most recent scan shows 100 congressional signals. Approximately 85 of them are $8,000 trades from a single representative (Gilbert Cisneros) spread across 60+ tickers. These are almost certainly a 10b5-1 diversification program or ETF-equivalent spread buying. Each individual signal has strength 0.02, which is economically meaningless. MIDGE is polluting the convergence engine with noise.

**Recommended minimum signal thresholds for economic viability:**

| Source | Minimum Trade Size for Signal Validity | Minimum Strength |
|--------|---------------------------------------|-----------------|
| Insider trade | $100,000 | 0.10 |
| Congressional trade | $50,000 | 0.10 |
| Contract award | $50M | 0.50 |
| Hiring signal | spike_ratio >= 2.0 | 0.40 |
| 8-K material (1.01, 1.03) | Any | 0.60 (direction-dependent) |

These filters would eliminate ~80% of the current congressional noise and focus the convergence engine on signals with genuine informational content.

### 7.2 The $8,000 Congressional Trade Diagnostic

The scan shows 85 of the 100 congressional signals from Gilbert Cisneros are all sales of $8,000 each across a wide spread of unrelated tickers. This is a characteristic pattern of a representative liquidating a diversified portfolio through a 10b5-1 plan. Every individual $8,000 sale is coded as "bearish" with strength 0.02.

These signals have zero informational content individually. Aggregated, they signal only "this person is selling a diversified portfolio" — which is economically neutral. The current convergence engine cannot detect this pattern and instead interprets them as 85 independent bearish signals across 60+ stocks.

**Required fix:** Congressional signal filtering should apply a minimum trade size of $50,000. Signals below this threshold should be stored for audit/discovery purposes but excluded from convergence calculations.

---

## 8. Leading vs. Lagging Indicators

This is the most strategically important classification. A lagging indicator tells you what already happened. A leading indicator tells you what is about to happen.

### 8.1 Genuinely Leading Signals (Information Not Yet Fully Priced In)

**1. Insider buying clusters (pre-earnings, pre-announcement)**
The cluster is formed by people with material non-public information. When 3+ C-suite executives buy within a 30-day window and the event has not yet occurred, this is a leading indicator. However, **after** a positive earnings surprise or material announcement, cluster buys that were predictive become lagging confirmations.

**2. Hiring blitzes correlated with active contract bids**
This is MIDGE's most genuinely leading signal. Job postings precede contract awards by 60-120 days. The market does not monitor job postings at the company level the way MIDGE does. This is differentiated alpha.

**3. Congressional trades from committee members in their own oversight sector**
The `politician_tracker.py` `CorrelationSignal` captures when a member of, say, the Armed Services Committee buys a defense stock before a major procurement cycle. This is genuine information asymmetry that may persist for weeks to months before becoming public.

**4. SAM.gov opportunities (early in competition period)**
When a large contract is posted and MIDGE identifies the most likely bidder based on historical patterns, this is a leading indicator for an announcement 3-18 months away.

### 8.2 Lagging Signals (Confirming What the Market Already Knows)

**1. Contract awards (USASpending, post-announcement)**
By the time a contract appears on USASpending, the announcement has already been made. For large caps, the market typically prices this in within 1-3 days. This is a lagging signal useful only for validating predictions and updating the Bayesian model, not for generating new entries.

**2. Most 8-K events (non-material items)**
8-K items like 7.01 (Reg FD Disclosure), 8.01 (Other Events), 9.01 (Financial Statements) are either already priced in or not directional enough to act on.

**3. Individual small insider trades from compensation plan exercises**
Option exercises and restricted stock vesting (Form 4 transaction codes M and F) are mechanically required sales that reveal nothing about the insider's view. These are currently being flagged — the scan shows high-value "bearish" Form 4 trades that are actually compensation executions. (NVDA's Colette Kress selling $1M+ across multiple days is almost certainly scheduled compensation plan execution, not informed bearish trading.)

**4. Congressional sales of diversified portfolios**
As analyzed above — portfolio rebalancing with no informational content.

### 8.3 The Insider Selling Problem in the Current Scan

The scan report shows a heavily bearish insider picture: NVDA, META, GOOGL, RTX, NOC, BA, AMZN all showing multiple high-strength bearish Form 4 signals. But reviewing the data:

- NVDA: Colette Kress (CFO) selling $600K-$3M in multiple transactions. This is almost certainly a Rule 10b5-1 scheduled plan.
- GOOGL: Sundar Pichai (CEO) selling $1.6M-$4.4M. Same pattern — likely scheduled.
- META, AMZN: Similar patterns.

**The problem:** MIDGE currently has no mechanism for distinguishing 10b5-1 plan sales (lagging, not informative) from discretionary sales (potentially informative). The `cluster_detector.py` correctly notes that transaction code F (tax withholding) should be filtered, but 10b5-1 plans are not being filtered.

**The domain "insider" is being dominated by lagging compensation execution, not leading information.** This explains why the insider domain is bearish across the board — almost entirely from scheduled compensation sales — while the actual fundamental picture for these tech stocks may be neutral to positive.

**Recommended fix:** Add `is_10b5_1_plan` detection to the Form 4 parser. The Form 4 XML contains a `planName` field when transactions are pursuant to a 10b5-1 plan. These should be stored but excluded from the bullish/bearish signal classification, or given a separate domain ("compensation") that is excluded from convergence calculations.

---

## 9. Specific Numbers: What Research Tells Us MIDGE's Alpha Is

Synthesizing academic research into concrete expectations for each signal type under ideal conditions:

| Signal Type | Expected Annual Alpha (ideal conditions) | Win Rate (approximate) | Optimal Holding Period |
|------------|------------------------------------------|----------------------|----------------------|
| Cluster insider buys (3+ insiders, C-suite) | 12-16% | 60-65% | 20-60 trading days |
| Individual CEO/CFO open-market purchase | 8-12% | 55-60% | 20-45 trading days |
| Congressional buy (committee member, own sector, >$100K) | 8-14% per quarter | 55-65% | 30-90 days from disclosure |
| Defense contractor hiring blitz + active bid | Unknown (insufficient study) | Estimated 55% | 60-120 days |
| Contract pre-announcement convergence (hiring + insider + bid) | Unknown | Estimated 60-70% | Until announcement +30 days |
| Contract award post-announcement (large cap) | 1-3% | 55% | 3-10 trading days |
| 8-K material event (bullish, e.g., 1.01) | Variable | 55-60% | 2-5 trading days |

---

## 10. What the Most Recent Scan Report Reveals About Timing

Analyzing the 2026-02-22 scan output with timing lens:

**Observation 1: The convergence alert fired on insider + institutional (RTX hiring).**
The RTX signal: bearish Form 4 (Eddy Shane G, DaSilva Kevin G, Williams Dantaya — all selling $1-3.5M each) PLUS bullish hiring (str=0.84). This creates a false bullish convergence because the insider selling is lagging compensation execution, not informed selling. The hiring blitz is a genuine pre-announcement signal. The convergence is noise.

**Observation 2: The scan window (168 hours = 7 days) is appropriate for this type of scan.**
The 7-day window correctly allows signals from the past week to accumulate before firing an alert. This is appropriate for the manual scan cadence.

**Observation 3: The domain status table shows all strengths as 0.00.**
This appears to be a bug in `write_report()`. The `domain_status` dictionary's strength field is being accessed as `status.get("avg_strength", 0)` but the actual field is `"strength"` (from `get_domain_status()`). The report table is showing zeros for all strengths, which means the domain status visualization is broken.

**Observation 4: 103 Form 4 signals, mostly bearish, from scheduled compensation plans.**
These dominate the convergence engine and push it to "bearish" for insider domain. They are economically meaningless as directional signals. Without filtering, they will consistently bias MIDGE bearish on any well-compensated, successful tech company — exactly backwards from what the sophisticated investor wants.

**Observation 5: The single genuinely interesting convergence in the scan is GD (General Dynamics).**
GD shows: bearish Form 4 (Rayha Mark selling $1.5M) AND bullish Form 4 (Rayha Mark buying $880K) AND neutral 8-K (2 events) AND bullish hiring (str=0.70). This multi-signal picture on a single stock is exactly what MIDGE should be surfacing — mixed insider signals with hiring bullish, suggesting complex transition. But MIDGE's convergence engine is not surfacing this because it works at the domain level, not the symbol level. The symbol-level convergence analysis is missing.

---

## Summary of Critical Timing Deficiencies and Recommended Fixes

**Priority 1 — Economic Noise Elimination:**
Filter congressional signals below $50,000 trade size from the convergence engine. This eliminates approximately 85% of the current congressional noise.

**Priority 2 — 10b5-1 Plan Detection:**
Add plan detection to Form 4 parsing. Exclude or domain-separate compensation plan executions. This will transform the insider domain from systematically bearish (due to scheduled selling) to properly directional.

**Priority 3 — Multi-Timeframe Convergence Windows:**
Implement separate Tier 1 (48h), Tier 2 (21d), and Tier 3 (90d) convergence analyzers. This is architecturally straightforward — instantiate three `ConvergenceAlerter` objects with different window parameters and different signal feeds.

**Priority 4 — Decay Rate Corrections:**
Update the decay rates as specified in Section 3. Most critically: slow down the ContractPrediction and SAM.gov opportunity decay (they're currently too fast), and speed up Form 8-K material event decay (currently too slow).

**Priority 5 — Scan Frequency Tiering:**
Implement EDGAR RSS polling every 2-4 hours. This is the highest-value latency improvement available — the difference between detecting an insider buy on the day of filing vs. the day after filing is the difference between capturing the fast-decay alpha or missing it entirely.

**Priority 6 — Symbol-Level Convergence:**
Add a secondary convergence pass at the symbol level: for each symbol with 3+ signals from different domains, generate a symbol-level convergence alert. This surfaces the GD-type situations (mixed insider signals with hiring confirmation) that are invisible in the current domain-only convergence.

**Priority 7 — Position Sizing:**
Add `position_size_pct` to `TradeSignal` using the fractional Kelly framework described in Section 4. Even a simple implementation (5-15% based on confidence tier) is vastly better than no sizing guidance.

---

*Sources consulted:*
- [Are Insider Trades Informative? — Lakonishok & Lee (2001)](https://www.semanticscholar.org/paper/Are-Insider-Trades-Informative-Lakonishok-Lee/7aa0b5e093d5791388de563d567bdc5543186e0f)
- [Profiting From Insider Transactions — 2iQ Research](https://www.2iqresearch.com/blog/profiting-from-insider-transactions-a-review-of-the-academic-research)
- [Do Insiders Cluster Trades With Colleagues — Alldredge (2019)](https://onlinelibrary.wiley.com/doi/abs/10.1111/jfir.12172)
- [STOCK Act Disclosure Rules — Nancy Pelosi Stock Tracker](https://nancypelosistocktracker.org/articles/disclosure-rules-explained)
- [Congressional Insider Trading Research — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9560883/)
- [Alpha Decay — Di Mascio, Lines, Naik (Journal of Finance)](https://jhfinance.web.unc.edu/wp-content/uploads/sites/12369/2016/02/Alpha-Decay.pdf)
- [Signal Decay Analysis — Microalphas](https://microalphas.com/signal-decay-patterns/)
- [SEC EDGAR API Documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [Insider Trading Transaction Codes — CorporateCounsel Blog](https://www.thecorporatecounsel.net/blog/2024/08/insider-trading-watch-your-form-4-transaction-codes.html)
- [Kelly Criterion Applications — QuantConnect](https://www.quantconnect.com/research/18312/kelly-criterion-applications-in-trading-systems/)
