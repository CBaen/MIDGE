# Lead Findings: Signal Architecture for Market Intelligence Integration
**Lens: Signal Architecture**
**Role: Lead (Phase 1)**
**Date: 2026-02-22**

---

## Executive Summary

After reading all 16 market module files, the EventBus, and the ApiGateway, the central finding is this: the existing modules already contain a near-complete signal taxonomy, but each module invented its own data shape. The work is not building something new — it is building the single membrane that every data shape collapses into before it reaches the intelligence layer.

The ConvergenceAlerter (`mae_core/market/intelligence/convergence_alerter.py`, line 133) already defines the required input: a `record_signal(signal_id, strength, domain, direction, confidence, velocity)` interface. Every other module in the stack must be designed to produce output that maps to those six parameters. That is the entire signal normalization problem.

---

## Part 1: What Each API Returns

### 1.1 SEC EDGAR (`mae_core/market/apis/sec_edgar/`)

**InsiderTrade** (models.py, line 76):
- Raw output: `filer_name`, `filer_title`, `filer_relationship`, `ticker_symbol`, `transaction_date`, `transaction_type` (A=acquire/D=dispose), `shares`, `price_per_share`, `total_value`, `shares_owned_after`
- Signal metadata already present on the dataclass: `signal_source = "insider"`, `decay_rate = 0.05` (14-day half-life), `confidence = 0.70` (implicit, not set on dataclass but assigned in parsing)
- **Direction mapping**: `transaction_type == "A"` → bullish; `"D"` → bearish
- **Strength mapping**: `total_value / normalization_constant` — currently no normalization exists

**Form8KEvent** (models.py, line 8):
- Raw output: `company_name`, `ticker_symbol`, `item_code`, `item_description`, `event_date`, `event_summary`, `material_impact` (bullish/bearish/neutral/unknown), `confidence = 0.70`
- `ITEM_CODES` dictionary (line 30) encodes a hand-coded directional bias for each item type — this is the edge detector's interpretation layer already built in
- **Direction mapping**: `material_impact` field is already populated by the parser (line 493)

### 1.2 Price Fetcher (`mae_core/market/apis/price_fetcher.py`)

**PriceData** (line 28):
- Returns: `symbol`, `price`, `timestamp`, `source` (yfinance/alpha_vantage), `open`, `high`, `low`, `volume`, `change_pct`
- This is NOT a signal — it is ground truth. The `change_pct` field is the outcome variable that all other signals must predict.
- **Critical architecture role**: `price_fetcher_for_outcomes()` (line 263) is already shaped as the feedback function for an outcome tracker. Price data closes the Bayesian loop.

### 1.3 House Stock Watcher (`mae_core/market/apis/house_stock_watcher.py`)

**CongressionalTrade** (line 43):
- Returns: `representative`, `party`, `district`, `ticker`, `transaction_type` (purchase/sale), `transaction_date`, `amount_low`, `amount_high`, `disclosure_date`
- Signal metadata: `signal_source = "congressional"`, `decay_rate = 0.03` (23-day half-life), `confidence = 0.65`
- **Critical gap**: There is a 45-day disclosure delay (noted in module docstring, line 10). The `disclosure_date` is NOT the trade date. `transaction_date` is. The signal's relevance must be measured from `transaction_date`, not when MIDGE sees it.
- **Direction mapping**: `"purchase" in transaction_type` → bullish; `"sale" in transaction_type` → bearish

### 1.4 Job Tracker (`mae_core/market/apis/job_tracker.py`)

**HiringSignal** (line 44):
- Returns: `company_name`, `ticker`, `jobs_24h`, `jobs_7d`, `jobs_30d`, `is_spike`, `spike_ratio`, `engineering_jobs`, `cleared_jobs`, `contract_related_jobs`, `confidence`
- Signal metadata: `signal_source = "hiring_tracker"`, `decay_rate = 0.07` (10-day half-life), `confidence = 0.5` (base, boosted to 0.90 max)
- **Direction**: Always bullish (hiring = growth expectation). No bearish signal possible from this source.
- **Strength mapping**: `spike_ratio` is the natural strength metric — already normalized (ratio, not absolute count)

### 1.5 USASpending (`mae_core/market/apis/usa_spending.py`)

**GovernmentContract** (line 35):
- Returns: `recipient_name`, `award_amount`, `award_date`, `award_type`, `description`, `naics_code`, `awarding_agency`
- Signal metadata: `signal_source = "contract"`, `decay_rate = 0.02` (35-day half-life)
- **Direction**: Always bullish for the recipient. No built-in confidence field — must be computed.
- **Ticker resolution gap**: `recipient_name` is a company name (e.g., "LOCKHEED MARTIN CORPORATION"), NOT a ticker. There is no ticker-to-company name mapping in this module. This mapping must be handled by the normalizer.

### 1.6 SAM.gov (`mae_core/market/apis/sam_gov.py`)

**ContractOpportunity** (line 43) and **ContractAward** (line 90):
- `ContractOpportunity` is a PRE-award solicitation — no winner yet, pure opportunity signal
- `ContractAward` is POST-award confirmation
- Neither has a `confidence` field
- Both lack ticker. `DEFENSE_NAICS` dictionary (line 274) maps NAICS codes to industry descriptions — useful for sector routing
- **Critical insight**: This is the only API that gives the BIDDING phase. Combining with JobTracker creates the leading indicator chain: SAM solicitation → hiring blitz → contract award.

---

## Part 2: What the Edge Detectors Expect as Input

### 2.1 ClusterDetector (`mae_core/market/edge/cluster_detector.py`)

**Input**: Queries Qdrant `midge_signals` collection directly (line 178). Expects stored payloads with fields:
- `signal_source = "sec_form4"` (line 183)
- `symbol` (line 184)
- `direction` (line 185)
- `details.filer_name`, `details.filer_title`, `details.filer_relationship` (lines 250-252)
- `details.shares`, `details.total_value`, `details.shares_owned_after` (lines 287-289)
- `details.transaction_code` (line 237)
- `details.transaction_date` (line 352)
- `timestamp` (line 200, used for date filtering)

**This is the Qdrant payload schema for Form 4 signals.** The normalizer must write these fields when storing insider trades.

### 2.2 FilingTimeAnalyzer (`mae_core/market/edge/filing_time_analyzer.py`)

**Input**: Queries Qdrant (line 197), expects:
- `ticker` (line 206)
- `signal_source` in `["insider", "sec_form8k"]` (line 207)
- `filing_date` (line 225) — must be ISO datetime or YYYY-MM-DD

**Output**: `FilingTimeSignal.confidence_modifier` (float, -0.15 to +0.10). This is a MODIFIER, not a primary signal — it adjusts the confidence of co-occurring Form 4 and 8-K signals.

### 2.3 PoliticianTracker (`mae_core/market/edge/politician_tracker.py`)

**Input**: Calls `get_recent_form4s()` (line 169) and `USASpendingClient.search_by_company()` (line 232) directly. Does NOT use Qdrant.
**Output**: `CorrelationSignal` with `confidence` (0.3 to 1.0), `correlation_type` ("politician_contract" or "insider_preannouncement"), `oversight_match` boolean.

**Critical gap**: `trade.shares_traded` is referenced (line 276) but `InsiderTrade` does not have a `shares_traded` attribute — it has `shares` (line 87 in models.py). This is a bug in the current code.

Also: `trade.is_purchase` (line 231 in contract_predictor.py) — `InsiderTrade` has no `is_purchase` property. It has `transaction_type` which must be checked as `== "A"`. Two places need fixing when normalizing.

### 2.4 ContractPredictor (`mae_core/market/edge/contract_predictor.py`)

**Input**: Calls `SAMGovClient`, `JobTracker`, and `get_recent_form4s()` directly. Uses Qdrant for historical wins (line 249).
**Output**: `ContractPrediction` with `confidence` (sum of component breakdowns), `predicted_winner`, `predicted_ticker`, `hiring_spike_ratio`, `insider_buy_value`.

This detector is the most horizontally integrated — it already IS a convergence detector for the defense sector specifically. It is a domain-specific version of what `ConvergenceAlerter` does generally.

---

## Part 3: What the Intelligence Layer Produces as Output

### 3.1 ConvergenceAlerter (`mae_core/market/intelligence/convergence_alerter.py`)

**The crown jewel.** Key design facts:

**Input interface** (line 133):
```python
alerter.record_signal(
    signal_id: str,      # Unique ID for this signal observation
    strength: float,     # 0-1 normalized signal strength
    domain: str,         # "insider", "congress", "contracts", "government", "technical", etc.
    direction: str,      # "bullish", "bearish", "neutral"
    confidence: float,   # Source reliability estimate 0-1
    velocity: float,     # Rate of change from VelocityDetector
    timestamp: datetime,
    metadata: dict       # Any additional context
)
```

**Domain categories** already defined (lines 115-129):
```python
"insider": "behavioral"
"congress": "behavioral"
"crypto": "market"
"technical": "market"
"volume": "market"
"sentiment": "social"
"reddit": "social"
"news": "information"
"events": "information"
"fundamentals": "financial"
"macro": "financial"
"government": "institutional"
"contracts": "institutional"
```

**Convergence logic** (line 220): Checks each direction separately. Collects signals with `strength >= min_strength` (default 0.6). Groups into `categories_seen`. Generates `ConvergenceAlert` when `len(domains_seen) >= min_domains` (default 2).

**Output** (`ConvergenceAlert`, line 44):
- `direction`, `strength`, `confidence`, `domains_converging`, `cross_domain_count`, `summary`, `urgency` (immediate/hours/days based on velocity)

**Critical insight about `get_actionable_summary()`** (line 362): Returns a recommendation dict — but it requires signals from AT LEAST 2 different categories (line 381). A single-category convergence (e.g., 3 insider signals) will return `"neutral"`. This is intentional — single-category signals stay as edge detector alerts, not convergence alerts.

### 3.2 ThompsonSampler (`mae_core/market/intelligence/thompson_sampler.py`)

**Input**: `update(signal_id, success: bool, regime: str)` — after observing an outcome.
**State**: Beta(alpha, beta) per (signal_id, regime). Seeded from `learning_config.LEARNING_CONFIG["source_reliability"]` (line 141). Initial seeds:
- `sec_edgar`: Beta(9.5, 0.5) — mean 0.95
- `form_4`: Beta(9.0, 1.0) — mean 0.90
- `capitol_trades`: Beta(8.5, 1.5) — mean 0.85
- `reddit`: Beta(3.0, 7.0) — mean 0.30

**Key method**: `sample(signal_id, regime)` (line 183) — samples from the distribution to get a reliability estimate. This is used at decision time to weight signals, not at ingest time.

**The regime parameter is currently unused in practice** — all distributions default to `"default"`. A future enhancement would feed market regime detection into Thompson sampling to maintain separate distributions per regime (bull/bear/sideways). This is architecturally present but not wired.

### 3.3 VelocityDetector (`mae_core/market/intelligence/velocity_detector.py`)

**Input**: `record(signal_id, value: float, timestamp)` — the value is an absolute metric (count of insider buys, dollar value of contracts, etc.)
**Output**: `VelocityState` with `current_velocity`, `current_acceleration`, `velocity_zscore`, `is_anomalous`, `is_accelerating`

**Critical role in the pipeline**: The `velocity` parameter in `ConvergenceAlerter.record_signal()` is meant to come from VelocityDetector. Currently there is no wiring — someone calling `record_signal()` manually passes `velocity=0.0`. The normalizer must bridge this: after computing velocity from VelocityDetector, pass it through to ConvergenceAlerter.

### 3.4 CorrelationTracker (`mae_core/market/intelligence/correlation_tracker.py`)

**Input**: `record(signal_id, value: float, timestamp, domain: str)` — values must be on the same scale (recommend 0-1) for Pearson correlation to be meaningful
**Output**: Correlation coefficients, anomaly detection (z-score > 2.5 threshold), cross-domain pair anomalies

**This is the general-purpose pattern discovery engine**. Unlike ConvergenceAlerter (which detects directional consensus), CorrelationTracker detects *structural relationships* — signals that move together regardless of whether that was expected. This is where novel cross-domain patterns emerge.

---

## Part 4: The EventBus — Channel Naming Patterns

### 4.1 How Existing Systems Use the EventBus

From `event_bus.py` (line 82): `publish(channel, message)` — message is serialized to JSON automatically for dicts.

From `register_callback(channel, callback)` (line 163): synchronous, called inline during `publish`.

**Observed channel naming convention** (from grep across codebase):
```
{subsystem}.{event_noun}
```
Examples:
- `cognition.decision_routed`
- `cognition.goal_update`
- `external.response_received`
- `external.request_failed`
- `pattern.advisory`
- `morphogenesis.spawn_request`
- `coordination.satiation_signal`
- `bootstrap.audit_complete`
- `substrate.topology_optimized`

Pattern: lowercase, dot-separated, subsystem prefix defines the publishing owner.

### 4.2 Triadic Connection Requirement (Law 1)

From `event_bus.py` lines 98-119: When `ConnectionRegistry` is sealed, every publish call is checked for triadic compliance. The source is extracted as `channel.split(".")[0]`. This means **the channel prefix IS the registered system name** — if you publish on `market.signal_detected`, the ConnectionRegistry looks for a registered system named `market`.

Every market EventBus channel requires a triadic connection registered with witnesses. Looking at how `external.py` bootstrap does this (lines 276-323): each channel gets `reg("source", "event_bus", eb, channel="...", witnesses=["w1", "w2"])`.

---

## Part 5: Existing ApiGateway Providers and Market Data

### 5.1 Registered Data Providers (from `bootstrap/external.py`)

| Provider | Env Var | Base URL | Data Type |
|----------|---------|----------|-----------|
| `marketaux` | `MAE_MARKETAUX_API_KEY` | `api.marketaux.com/v1` | Financial news + sentiment |
| `finnhub` | `MAE_FINNHUB_API_KEY` | `finnhub.io/api/v1` | Real-time quotes, candles, company data |
| `alphavantage` | `MAE_ALPHAVANTAGE_API_KEY` | `alphavantage.co` | Time series, crypto, forex |
| `tavily` | `MAE_TAVILY_API_KEY` | Tavily search | Web search with AI answers |

### 5.2 What These Provide That Market Modules Don't

- **MarketAux**: News articles with sentiment scores — maps to `domain="news"`, `domain="sentiment"` in ConvergenceAlerter
- **Finnhub**: Real-time price data — can replace/augment yfinance for `domain="technical"` (price momentum, volume signals)
- **Alpha Vantage**: Historical time series needed for VelocityDetector baseline (longer history than yfinance's casual use)
- **Tavily**: General web search — the "unknown source" gateway. LLM analysis of search results maps to `domain="sentiment"` or `domain="news"` with lower confidence weighting (~0.55)

### 5.3 How ApiGateway Providers Join the Signal Mesh

ApiGateway uses the `external.response_received` channel (line 47, `CH_EXTERNAL_RESPONSE`). Market signal adapters must subscribe to this channel and, when a financial provider response arrives, parse it and call the appropriate normalizer. The flow:

```
Agent submits ApiRequest with provider="finnhub"
→ ApiGateway executes HTTP call
→ Publishes response on "external.response_received"
→ MarketSignalAdapter callback parses response
→ Converts to normalized MarketSignal
→ Publishes on "market.signal.raw"
→ SignalNormalizer converts to ConvergenceAlerter.record_signal() call
```

---

## Part 6: The Normalized MarketSignal Dataclass

### 6.1 Design Principles

The normalizer must accept ALL data shapes and reduce them to a single common format. The design is constrained by what ConvergenceAlerter (line 133), VelocityDetector (line 99), and CorrelationTracker (line 101) each need.

The minimum required fields that serve ALL three consumers:

```python
@dataclass
class MarketSignal:
    # Identity
    signal_id: str          # UUID or "{source}:{symbol}:{timestamp}"
    source: str             # "sec_edgar", "congress", "hiring_tracker", etc. — maps to ThompsonSampler signal_id
    symbol: str             # Ticker (AAPL, BTC-USD, etc.) — can be "" for macro signals
    asset_class: str        # "stock", "crypto", "futures", "commodities", "macro"

    # Classification (ConvergenceAlerter needs these)
    domain: str             # "insider", "congress", "contracts", "government", "technical", "sentiment", "news"
    direction: str          # "bullish", "bearish", "neutral"
    strength: float         # 0.0-1.0 normalized intensity

    # Reliability (ThompsonSampler needs this as key)
    confidence: float       # Source reliability estimate 0.0-1.0
    decay_rate: float       # Per-day decay (from learning_config.decay_rates)

    # Time (VelocityDetector needs timestamp series)
    timestamp: datetime     # When the underlying event occurred (NOT when MIDGE received it)
    received_at: datetime   # When MIDGE received/detected this signal

    # Velocity (ConvergenceAlerter velocity parameter)
    velocity: float         # Populated by VelocityDetector after recording; default 0.0

    # Ground truth feedback loop
    outcome_symbol: str     # Ticker to check for price outcome (often == symbol)
    outcome_window_days: int  # How many days forward to measure outcome

    # Raw payload (for audit and re-processing)
    raw_type: str           # "InsiderTrade", "Form8KEvent", "CongressionalTrade", etc.
    raw_payload: dict       # Original dataclass as dict (for edge detector queries)

    # Context (optional, for pattern discovery)
    metadata: dict          # Extra fields (sector, committee name, NAICS code, etc.)
```

### 6.2 Source-to-Domain Mapping Table

| Source Module | `source` value | `domain` | Typical `strength` derivation |
|---------------|---------------|----------|-------------------------------|
| InsiderTrade (buy) | `"sec_form4"` | `"insider"` | `min(1.0, total_value / 1_000_000)` |
| InsiderTrade (sell) | `"sec_form4"` | `"insider"` | `min(1.0, total_value / 500_000)` |
| Form8KEvent (bearish item) | `"sec_form8k"` | `"events"` | `confidence` field (0.50-0.70) |
| Form8KEvent (bullish item) | `"sec_form8k"` | `"events"` | `confidence` field |
| CongressionalTrade | `"congressional"` | `"congress"` | `min(1.0, amount_high / 500_000)` |
| HiringSignal | `"hiring_tracker"` | `"institutional"` | `min(1.0, spike_ratio / 5.0)` |
| GovernmentContract | `"contract_award"` | `"contracts"` | `min(1.0, award_amount / 100_000_000)` |
| ContractOpportunity | `"sam_gov"` | `"contracts"` | `0.3` (opportunity only, no confirmed winner) |
| ClusterSignal | `"insider_cluster"` | `"insider"` | `confidence` field |
| ContractPrediction | `"contract_prediction"` | `"contracts"` | `confidence` field |
| MarketAux news | `"marketaux"` | `"news"` | sentiment score (0-1) |
| Finnhub quote | `"finnhub"` | `"technical"` | `abs(change_pct) / 10.0` |
| Alpha Vantage | `"alphavantage"` | `"technical"` | computed from time series |
| Tavily search | `"tavily"` | `"sentiment"` | relevance score |
| LLM analysis | `"llm_analysis"` | `"sentiment"` | 0.5-0.7 (use source reliability) |

### 6.3 Strength Normalization Rationale

Each source domain has a different natural scale. The normalizer must apply domain-specific scaling:

- **Dollar values** (insider trades, congressional trades, contracts): log-scale normalization or clamp-at-threshold. A $1M insider buy is high signal. A $100M contract is high signal. $10 contract = 0.1 signal. Use: `min(1.0, value / threshold)` where threshold is the "maximum meaningful" value for that domain.
- **Ratios** (hiring spike_ratio): `min(1.0, ratio / 5.0)` — a 5x spike = maximum signal, anything above is still capped at 1.0
- **Counts** (number of insiders in cluster): `min(1.0, count / 5.0)` — 5+ insiders = maximum
- **Percentages** (price change_pct): `min(1.0, abs(change_pct) / 10.0)` — 10% move = maximum

---

## Part 7: EventBus Channel Architecture

### 7.1 Market Signal Channels

Following the existing `{subsystem}.{event_noun}` convention:

```
# Ingest layer — raw signals as they arrive from data sources
market.signal.raw            # Any normalized MarketSignal before velocity/correlation
market.signal.scored         # MarketSignal after Thompson scoring applied
market.signal.decayed        # Periodic decay update events

# Edge detector outputs
market.edge.cluster_detected     # ClusterSignal published
market.edge.correlation_found    # CorrelationSignal (politician+contract)
market.edge.filing_anomaly       # FilingTimeSignal with suspicious timing
market.edge.contract_predicted   # ContractPrediction published

# Intelligence layer outputs
market.intel.velocity_anomaly    # VelocityDetector flags anomalous velocity
market.intel.correlation_anomaly # CorrelationTracker flags cross-domain anomaly
market.intel.convergence         # ConvergenceAlerter fires ConvergenceAlert
market.intel.actionable          # Final actionable summary with recommendation

# Feedback loop
market.outcome.price_update      # Price data for Bayesian update
market.outcome.prediction_result # Outcome observed — feeds ThompsonSampler.update()
```

### 7.2 Channel-to-Consumer Map

| Channel | Published By | Consumed By |
|---------|-------------|-------------|
| `market.signal.raw` | API fetchers / ApiGateway adapter | VelocityDetector, CorrelationTracker, ThompsonSampler |
| `market.signal.scored` | SignalNormalizer | ConvergenceAlerter |
| `market.edge.cluster_detected` | ClusterDetector | ConvergenceAlerter, Qdrant writer |
| `market.edge.correlation_found` | PoliticianTracker | ConvergenceAlerter |
| `market.edge.contract_predicted` | ContractPredictor | ConvergenceAlerter |
| `market.intel.convergence` | ConvergenceAlerter | Agent decision cascade, Qdrant writer |
| `market.intel.actionable` | ConvergenceAlerter | Mae's cognition layer (decision_router) |
| `market.outcome.price_update` | PriceFetcher scheduler | OutcomeTracker, ThompsonSampler |
| `market.outcome.prediction_result` | OutcomeTracker | ThompsonSampler |

### 7.3 Channel Architecture for General-Purpose Pattern Discovery

The ConvergenceAlerter already handles known patterns via its domain categories. For *general* pattern discovery, the CorrelationTracker is the mechanism — but it needs to be exposed as a subscriber on `market.signal.raw` and its outputs need to flow back into the system.

The key design principle: **the channel is the category, not the signal**. Any data source — weather API, social media scraper, LLM analysis — can publish to `market.signal.raw` with an appropriate `domain` field. The CorrelationTracker and ConvergenceAlerter do not care what the source is; they care only about `domain`, `direction`, `strength`, and `timestamp`.

This means adding a new data source is a one-file operation: write an adapter that maps the source's output to `MarketSignal` and publishes on `market.signal.raw`. No changes to the intelligence layer.

### 7.4 The Discovery Stream

For genuinely novel pattern discovery (the "find correlations we haven't programmed"):
```
market.discovery.anomaly_correlation   # Unusual pair from CorrelationTracker
market.discovery.new_pattern           # Pattern not seen in thompson_distributions.json
market.discovery.regime_shift          # Velocity divergence across multiple domains
```

These feed into Qdrant's `midge_signals` collection with a `signal_source = "discovery"` tag, creating a growing library of discovered patterns. Over time, ThompsonSampler adds Beta distributions for novel signal IDs as they prove their predictive value.

---

## Part 8: Price Data as Ground Truth Feedback Loop

### 8.1 The Feedback Loop Design

The existing `PriceFetcher.get_historical_price()` (line 92) and `price_fetcher_for_outcomes()` (line 263) are explicitly shaped for this. The loop:

1. **Signal detected** → `MarketSignal` created with `outcome_symbol` and `outcome_window_days`
2. **Prediction recorded** → stored in `data/market/predictions.jsonl`
3. **N days pass** → OutcomeTracker fires (scheduled step hook)
4. **Price fetched** → `PriceFetcher.get_historical_price(outcome_symbol, prediction_date + N_days)`
5. **Outcome computed** → did price move in predicted direction by >2%? (threshold TBD)
6. **ThompsonSampler.update(signal_id, success, regime)** called
7. **Beta distribution shifts** → source reliability is updated
8. **`market.outcome.prediction_result` published** on EventBus

### 8.2 Multi-Asset Classes

The current PriceFetcher uses yfinance which supports:
- Stocks: `"AAPL"`, `"MSFT"`
- Crypto: `"BTC-USD"`, `"ETH-USD"` (yfinance format)
- ETFs and indices: `"SPY"`, `"QQQ"`, `"GLD"` (commodities proxy)
- Futures: `"ES=F"` (S&P), `"CL=F"` (crude oil), `"GC=F"` (gold)

The `asset_class` field in `MarketSignal` routes to the right price format. No separate price fetcher needed — yfinance handles all of these.

---

## Part 9: The TradeSignal Output Format

This is what the system ultimately produces — the actionable buy/sell recommendation.

```python
@dataclass
class TradeSignal:
    # Identity
    trade_id: str                  # UUID
    timestamp: datetime            # When generated

    # Target
    symbol: str                    # "AAPL", "BTC-USD", "GC=F"
    asset_class: str               # stock/crypto/futures/commodities

    # Recommendation
    direction: str                 # "BUY", "SELL", "HOLD"
    confidence: float              # 0.0-1.0 (from ConvergenceAlerter)
    strength: float                # 0.0-1.0 (signal intensity)
    urgency: str                   # "immediate" / "hours" / "days" (from alert.urgency)

    # Evidence
    domains_converging: List[str]  # Which domains agree
    cross_domain_count: int        # Number of independent category types
    signal_count: int              # Total signals contributing

    # Bayesian weighting
    thompson_weighted_confidence: float  # confidence adjusted by ThompsonSampler sampling
    regime: str                    # "default" / "bull" / "bear" / "sideways"

    # Reasoning (for audit trail)
    summary: str                   # ConvergenceAlert.summary text
    primary_signal_ids: List[str]  # Top signals driving this recommendation

    # Risk context (not yet implemented — future work)
    expected_move_pct: float       # Model's expected price move
    expected_window_days: int      # Within how many days

    # Lifecycle
    expires_at: datetime           # timestamp + (decay rate applied to convergence window)
    outcome_symbol: str            # Which price to check for outcome
    prediction_id: str             # Links to predictions.jsonl entry
```

### 9.1 How TradeSignal is Generated

`TradeSignal` is derived from `ConvergenceAlert` (convergence_alerter.py line 44) after Thompson adjustment:

1. `ConvergenceAlerter.check_convergence()` returns a `ConvergenceAlert`
2. For each signal in `alert.signals`, sample `ThompsonSampler.sample(signal.signal_id)`
3. Weight the alert's confidence by the geometric mean of Thompson samples
4. If weighted confidence > threshold (e.g., 0.65), emit `TradeSignal`
5. Direction: `alert.direction` → "BUY" (bullish) / "SELL" (bearish) / "HOLD" (neutral)
6. Publish on `market.intel.actionable` EventBus channel

---

## Part 10: Known Gaps and Design Risks

### 10.1 Bug: Missing InsiderTrade Attributes

In `contract_predictor.py` line 231: `trade.is_purchase` — `InsiderTrade` has no `is_purchase` property. Correct check is `trade.transaction_type == "A"`.

In `politician_tracker.py` line 276: `trade.shares_traded` — `InsiderTrade` has `shares` not `shares_traded`.

These are live bugs that will raise `AttributeError` when those code paths execute.

### 10.2 Gap: Ticker Resolution

GovernmentContract (`usa_spending.py`, line 35) and ContractOpportunity (`sam_gov.py`, line 43) have no ticker field. A `TickerResolver` service (company_name → ticker) is needed. Options: maintain a static mapping dict (current partial approach in `politician_tracker.py` lines 365-381), use yfinance `yf.Ticker(company_name).info`, or use a dedicated company-to-ticker database. This is required before these modules can populate `MarketSignal.symbol`.

### 10.3 Gap: VelocityDetector is Disconnected from ConvergenceAlerter

The `velocity` parameter in `ConvergenceAlerter.record_signal()` is always 0.0 in current practice. The `urgency` calculation (lines 258-265) uses velocity to classify as immediate/hours/days. Without real velocity values, all alerts will be classified as `urgency="days"`. The normalizer must bridge VelocityDetector → ConvergenceAlerter.

### 10.4 Gap: No Qdrant Schema

`ClusterDetector._query_recent_trades()` (line 172) queries Qdrant with specific payload field expectations. These fields come from how signals are stored — but there is no storage code that creates this schema. The `store_cluster_signal()` function (line 588) stores cluster signals but not the raw Form 4 trades that ClusterDetector reads. A `SignalWriter` service must define and maintain the Qdrant schema.

### 10.5 Risk: Congressional Trade Timing

The 45-day STOCK Act disclosure delay means `CongressionalTrade.transaction_date` can be up to 45 days before `disclosure_date`. When computing signal age for decay purposes, use `transaction_date`, not `disclosure_date`. Otherwise a 45-day-old trade appears fresh.

### 10.6 Risk: Single-Category Convergence

ConvergenceAlerter's `get_actionable_summary()` requires signals from at least 2 different *categories* (behavioral, market, institutional, etc.) to issue a non-neutral recommendation. With only government/institutional data sources active (contracts + hiring), ALL signals fall into the `"institutional"` category and convergence will always return `"neutral"`. MIDGE needs at least one price/technical signal source active to form actionable cross-category recommendations.

---

## Part 11: Complete Data Flow Architecture

```
[Data Sources]                [Signal Normalizer]         [EventBus: market.signal.raw]
SEC EDGAR Form4  ──────────→  InsiderTradeAdapter  ──────→
SEC EDGAR Form8K ──────────→  Form8KAdapter        ──────→
HouseStockWatcher ─────────→  CongressAdapter      ──────→   VelocityDetector
JobTracker  ───────────────→  HiringAdapter        ──────→   CorrelationTracker
USASpending  ──────────────→  ContractAdapter      ──────→   ThompsonSampler
SAM.gov  ──────────────────→  SolicationAdapter    ──────→
ApiGateway(marketaux) ─────→  NewsAdapter          ──────→
ApiGateway(finnhub) ───────→  PriceAdapter         ──────→  [market.signal.scored]
ApiGateway(tavily) ────────→  SearchAdapter        ──────→
                                                             ↓
[Edge Detectors]                                    ConvergenceAlerter
ClusterDetector ──────────→  market.edge.cluster_detected →
PoliticianTracker ─────────→  market.edge.correlation_found →
FilingTimeAnalyzer ────────→  (modifier, not primary signal)
ContractPredictor ─────────→  market.edge.contract_predicted →
                                                             ↓
[Intelligence]                                      [market.intel.convergence]
VelocityDetector ─────────→  market.intel.velocity_anomaly →
CorrelationTracker ────────→  market.intel.correlation_anomaly →
ConvergenceAlerter ────────→  market.intel.actionable ────→
                                                             ↓
[Output]                                            TradeSignal
ThompsonSampler (weighting) ─────────────────────→  (buy/sell/hold recommendation)
                                                             ↓
[Feedback Loop]                                     predictions.jsonl
PriceFetcher (outcome check) ─────────────────────→ outcomes.jsonl
                                                    ThompsonSampler.update()
```

---

## References

All file path references are absolute to `C:\Users\baenb\projects\MIDGE\`:

- `mae_core/market/apis/sec_edgar/models.py` — InsiderTrade, Form8KEvent dataclasses
- `mae_core/market/apis/sec_edgar/client.py` — SECEdgarClient, rate limiting, XML parsing
- `mae_core/market/apis/price_fetcher.py` — PriceData, PriceFetcher, outcome feedback function
- `mae_core/market/apis/house_stock_watcher.py` — CongressionalTrade, 45-day disclosure lag
- `mae_core/market/apis/job_tracker.py` — HiringSignal, spike detection
- `mae_core/market/apis/usa_spending.py` — GovernmentContract, AGENCY_COMMITTEE_MAP
- `mae_core/market/apis/sam_gov.py` — ContractOpportunity, ContractAward, DEFENSE_NAICS
- `mae_core/market/edge/cluster_detector.py` — Qdrant payload schema (lines 183-189)
- `mae_core/market/edge/politician_tracker.py` — CorrelationSignal, `shares_traded` bug (line 276)
- `mae_core/market/edge/filing_time_analyzer.py` — FilingTimeSignal, confidence modifier
- `mae_core/market/edge/contract_predictor.py` — ContractPrediction, `is_purchase` bug (line 231)
- `mae_core/market/intelligence/convergence_alerter.py` — ConvergenceAlerter.record_signal() (line 133), domain_categories (line 115), ConvergenceAlert (line 44)
- `mae_core/market/intelligence/thompson_sampler.py` — BetaDistribution, update() (line 197), DATA_DIR path (line 27)
- `mae_core/market/intelligence/velocity_detector.py` — VelocityDetector.record() (line 99), VelocityState
- `mae_core/market/intelligence/correlation_tracker.py` — CorrelationTracker, detect_cross_domain_anomalies()
- `mae_core/market/intelligence/learning_config.py` — source_reliability dict (line 36), decay_rates (line 22)
- `mae_core/backbone/event_bus.py` — publish() (line 82), channel naming, ConnectionRegistry integration (line 98)
- `mae_core/external/api_gateway.py` — registered channels (lines 46-50), response flow
- `mae_core/bootstrap/external.py` — provider registration (lines 207-251), channel naming (lines 276-323)
