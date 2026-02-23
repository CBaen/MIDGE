# MIDGE Data Schema

Canonical reference for all data structures flowing through MIDGE's pipeline. Every signal, prediction, outcome, and learned distribution follows these schemas.

---

## 1. MarketSignal (Core Data Type)

The normalized format for ALL market data. Every source adapter converts raw data into this shape before it enters the convergence engine.

**Source:** `mae_core/market/signal.py`
**Stored in:** `data/midge/signals/{date}.jsonl`

| Field | Type | Description |
|-------|------|-------------|
| `signal_id` | string | `"{source}:{symbol}:{date}"` or UUID |
| `source` | string | Origin: `sec_form4`, `sec_form8k`, `congressional`, `hiring_tracker`, `contract_award`, `sam_gov`, `insider_cluster`, `contract_prediction`, `politician_correlation` |
| `symbol` | string | Ticker symbol. Empty `""` for macro/pre-ticker signals |
| `asset_class` | string | `"stock"` (future: `"crypto"`, `"futures"`, `"commodities"`) |
| `domain` | string | Classification bucket: `insider`, `congress`, `contracts`, `government`, `institutional`, `events`, `institutional_synthesis` |
| `direction` | string | `"bullish"`, `"bearish"`, `"neutral"` |
| `strength` | float | 0.0-1.0 normalized. Log-linear for insider trades |
| `confidence` | float | 0.0-1.0 source reliability estimate |
| `decay_rate` | float | Per-day exponential decay (calibrated per source, see Section 7) |
| `velocity` | float | Per-day rate of change (populated by VelocityDetector) |
| `timestamp` | ISO 8601 | When the underlying event occurred |
| `received_at` | ISO 8601 | When MIDGE detected/received the signal |
| `outcome_symbol` | string | Ticker to check for price outcome |
| `outcome_window_days` | int | Days forward to evaluate (per-type, see Section 6) |
| `raw_id` | string | Original record ID in source system |
| `raw_type` | string | Source dataclass name (e.g., `"InsiderTrade"`) |
| `metadata` | object | Source-specific fields (varies by source, see Section 2) |

---

## 2. Source-Specific Metadata

Each `source` type carries different fields in the `metadata` object.

### sec_form4 (Insider Trades)

| Field | Type | Description |
|-------|------|-------------|
| `filer_name` | string | Insider's name (e.g., "Pichai Sundar") |
| `filer_title` | string | Role/title |
| `transaction_type` | string | `"A"` (acquired), `"D"` (disposed) |
| `shares` | float | Number of shares |
| `price_per_share` | float | Price per share |
| `total_value` | float | Total transaction value |
| `company_name` | string | Company name |

### sec_form8k (Material Events)

| Field | Type | Description |
|-------|------|-------------|
| `item_code` | string | SEC item code (e.g., "1.01", "5.02") |
| `item_description` | string | Human-readable event type |
| `event_summary` | string | Filing context (first 200 chars) |
| `material_impact` | string | `"bullish"`, `"bearish"`, `"neutral"`, `"unknown"` |
| `company_name` | string | Company name |

### congressional (Congress Member Trades)

| Field | Type | Description |
|-------|------|-------------|
| `representative` | string | Politician name |
| `party` | string | `"D"`, `"R"`, `""` |
| `district` | string | Office/state |
| `transaction_type` | string | `"purchase"`, `"sale"`, `"sale_partial"` |
| `amount_range` | string | Disclosed range (e.g., "$50,001 - $100,000") |
| `amount_low` | float | Lower bound |
| `amount_high` | float | Upper bound |
| `owner` | string | `"Self"`, `"Spouse"`, `"Joint"`, `"Child"` |
| `asset_description` | string | Asset name if no ticker |

### hiring_tracker (Job Market Spikes)

| Field | Type | Description |
|-------|------|-------------|
| `company_name` | string | Employer |
| `jobs_24h` | int | New postings in 24h |
| `jobs_7d` | int | New postings in 7d |
| `jobs_30d` | int | New postings in 30d |
| `spike_ratio` | float | Current vs baseline ratio |
| `is_spike` | bool | Above threshold? |
| `engineering_jobs` | int | Engineering-specific postings |
| `cleared_jobs` | int | Security-clearance postings |
| `contract_related_jobs` | int | Contract-specific postings |

### contract_award (Government Contracts)

| Field | Type | Description |
|-------|------|-------------|
| `recipient_name` | string | Awardee company |
| `award_amount` | float | Dollar value |
| `award_type` | string | Contract type |
| `awarding_agency` | string | Federal agency |
| `description` | string | Contract description |
| `naics_code` | string | Industry code |
| `start_date` | string | Performance start |
| `end_date` | string | Performance end |

### sam_gov (Contract Opportunities)

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Opportunity title |
| `department` | string | Federal department |
| `agency` | string | Specific agency |
| `naics_code` | string | Industry code |
| `estimated_value` | float | Estimated contract value |
| `response_deadline` | string | Bid deadline |
| `contract_type` | string | Contract type |
| `url` | string | SAM.gov listing URL |

### insider_cluster (Cluster Detector Output)

| Field | Type | Description |
|-------|------|-------------|
| `insider_count` | int | Number of insiders in cluster |
| `total_value` | float | Combined purchase value |
| `weighted_score` | float | Conviction-weighted score |
| `avg_conviction` | float | Average insider conviction |
| `has_csuite` | bool | C-suite participant? |
| `window_days` | int | Cluster time window |

### contract_prediction (Pre-Announcement Prediction)

| Field | Type | Description |
|-------|------|-------------|
| `predicted_winner` | string | Company name |
| `contract_title` | string | Contract being predicted |
| `contract_value` | float | Estimated value |
| `hiring_blitz_detected` | bool | Hiring spike found? |
| `hiring_spike_ratio` | float | Spike magnitude |
| `insider_buying_detected` | bool | Insider buying found? |
| `insider_buy_value` | float | Insider purchase amount |
| `confidence_breakdown` | object | Per-factor confidence |
| `contract_deadline` | string | Expected award date |
| `expected_award_date` | string | Predicted announcement |

---

## 3. Prediction Record

Registered when a signal qualifies for outcome tracking. Evaluated after the outcome window elapses.

**Stored in:** `data/market/predictions.jsonl`

### Current Format (post-Phase D)

| Field | Type | Description |
|-------|------|-------------|
| `signal_id` | string | UUID |
| `source` | string | Signal source type |
| `symbol` | string | Ticker |
| `direction` | string | `"up"`, `"down"`, `""` |
| `timestamp` | ISO 8601 | When prediction was made |
| `outcome_window_days` | int | Days to wait before evaluating |
| `outcome_symbol` | string | Ticker to check (optional override) |
| `metadata` | object | Optional extra context |

### Legacy Format (pre-Phase D)

| Field | Type | Description |
|-------|------|-------------|
| `prediction_id` | string | UUID |
| `symbol` | string | Ticker |
| `direction` | string | `"bullish"`, `"bearish"` |
| `confidence` | float | Signal confidence |
| `entry_price` | float | Price at prediction time |
| `predicted_at` | ISO 8601 | Timestamp (NOTE: `predicted_at` not `timestamp`) |
| `outcome_due` | ISO 8601 | When to evaluate |
| `timeframe` | string | `"1d"`, etc. |
| `prediction_source` | string | `"midge"` |

**Compatibility:** OutcomeTracker handles both formats via `pred.get("timestamp") or pred.get("predicted_at")`.

---

## 4. Outcome Record

Written after a prediction is evaluated against actual price movement.

**Stored in:** `data/market/outcomes.jsonl`

| Field | Type | Description |
|-------|------|-------------|
| `signal_id` | string | Matches prediction's signal_id |
| `source` | string | Signal source type |
| `symbol` | string | Ticker |
| `direction` | string | Predicted direction |
| `predicted_at` | ISO 8601 | When prediction was made |
| `evaluated_at` | ISO 8601 | When outcome was checked |
| `window_days` | int | Outcome window |
| `price_change_pct` | float | Actual price movement (positive = gain) |
| `success` | bool | Did magnitude + direction match? |
| `min_move_threshold` | float | Success threshold used (currently 5.0%) |

---

## 5. Thompson Distribution (Bayesian Brain)

Learned reliability estimates per signal source. Updated by the outcome feedback loop.

**Stored in:** `data/market/thompson_distributions.json`

```json
{
  "source_name": {
    "regime_name": {
      "alpha": float,  // Beta distribution alpha (successes + 1)
      "beta": float    // Beta distribution beta (failures + 1)
    }
  }
}
```

**Interpretation:** `alpha / (alpha + beta)` = estimated success probability. Higher alpha = more reliable source. Default prior: `Beta(1, 1)` (uniform).

**Regime keys:** `"default"` (standard), future: `"bull"`, `"bear"`, `"volatile"`, `"sideways"`.

### Thompson History Record

**Stored in:** `data/market/thompson_history.jsonl`

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | ISO 8601 | When update occurred |
| `signal_id` | string | Source type updated |
| `success` | bool | Was the prediction correct? |
| `regime` | string | Market regime at evaluation time |
| `old_alpha` | float | Previous alpha |
| `old_beta` | float | Previous beta |
| `new_alpha` | float | Updated alpha |
| `new_beta` | float | Updated beta |
| `old_mean` | float | Previous success probability |
| `new_mean` | float | Updated success probability |

---

## 6. Outcome Windows by Signal Type

Each signal type has a calibrated outcome window (how long to wait before checking if the prediction was correct).

| Source | Window (days) | Rationale |
|--------|---------------|-----------|
| `sec_form4` | 45 | Insider trades lead by 20-60 days |
| `insider_cluster` | 60 | Clusters are longer-horizon signals |
| `congressional` | 14 | Already stale by disclosure; 14d from disclosure |
| `sec_form8k` | 5 | Market prices binary events within days |
| `contract_prediction` | 90 | Pre-announcement, long lead time |
| `contract_award` | 45 | Post-announcement drift |
| `hiring_tracker` | 90 | Hiring leads contracts by 60-120 days |
| `sam_gov` | 90 | Competition periods last months |
| `correlation` | 21 | Moderate-term signal |

---

## 7. Decay Rates (Signal Staleness)

Per-day exponential decay rates. Signal effective strength = `strength * exp(-decay_rate * age_days)`.

| Source | Decay Rate | Half-Life (days) | Citation |
|--------|-----------|-------------------|----------|
| `sec_form4` | 0.035 | ~20 | Lakonishok & Lee 2001 |
| `insider_cluster` | 0.025 | ~28 | Alldredge 2019 |
| `congressional` | 0.05 | ~14 | Disclosure delay |
| `sec_form8k` | 0.25 | ~3 | Rapid market pricing |
| `contract_prediction` | 0.018 | ~39 | Pre-announcement lead |
| `contract_award` | 0.07 | ~10 | Post-announcement drift |
| `hiring_tracker` | 0.015 | ~46 | Long lead to contracts |
| `sam_gov` | 0.008 | ~87 | Competition periods |
| `correlation` | 0.04 | ~17 | Combined staleness |

---

## 8. Supporting Data Files

### Registered Signals (`data/market/registered_signals.json`)

JSON array of signal_id strings already submitted to the outcome tracker. Prevents double-registration across scan runs.

```json
["sec_form4:MSFT:02/18/2026", "congressional:RY:2026-01-07", ...]
```

### Discovery Log (`data/market/discovery_log.jsonl`)

Records of autonomous pattern discovery cycles.

| Field | Type | Description |
|-------|------|-------------|
| `cycle_id` | string | Unique cycle identifier |
| `timestamp` | ISO 8601 | When discovery ran |
| `topic` | string | Discovery topic |
| `rounds` | int | Number of exploration rounds |
| `patterns_found` | int | Patterns detected |
| `patterns_selected` | int | Patterns chosen for testing |
| `predictions_made` | int | New predictions generated |
| `summary` | string | Human-readable summary |

### Config History (`data/market/config_history.jsonl`)

Learning parameter evolution (LearningConfig self-modifications).

### Watchlist (`data/midge/watchlist.json`)

Scan targets.

```json
{
  "tickers": ["AAPL", "MSFT", ...],
  "keywords": ["cybersecurity", "defense", ...],
  "companies": {"Lockheed Martin": "LMT", ...}
}
```

### Scan Reports (`data/midge/scans/{date}-{time}.md`)

Markdown intelligence reports. One per scan run. Contains: executive summary, convergence alerts, per-ticker convergence, multi-timeframe convergence, signal counts, domain status, signals by symbol, outcome tracking stats, price snapshot.

### Signal Archives (`data/midge/signals/{date}.jsonl`)

Raw MarketSignal records (one JSON object per line). Cold storage backup — always written regardless of Qdrant availability.

---

## 9. Convergence Alert (Runtime Only)

Generated by ConvergenceAlerter during analysis phase. Not persisted to disk (appears in scan reports).

| Field | Type | Description |
|-------|------|-------------|
| `direction` | string | `"BULLISH"`, `"BEARISH"` |
| `strength` | float | 0.0-1.0 convergence strength |
| `confidence` | float | 0.0-1.0 based on domain count + strength |
| `categories` | list[string] | Domains contributing to convergence |
| `signals` | int | Total signal count across domains |
| `urgency` | string | `"immediate"`, `"hours"`, `"days"` |
| `description` | string | Human-readable summary |

### Per-Ticker Convergence

Same shape as above, but scoped to a single symbol. Generated by `check_ticker_convergence(min_domains=2)`.

### Multi-Timeframe Convergence

Three tiers of ConvergenceAlerter, each with different signal routing:

| Tier | Name | Window | Sources |
|------|------|--------|---------|
| 1 | Tactical | 48h | `sec_form4`, `sec_form8k` |
| 2 | Strategic | 21d | `congressional`, `contract`, `insider_cluster`, `correlation` |
| 3 | Thematic | 90d | `sam_gov`, `hiring_tracker`, `contract_prediction` |

**Cross-tier detection:** When the same ticker appears in 2+ tiers, confidence is multiplied by 0.7 (Alpha's independence amendment).

---

## 10. InsiderTrade (SEC EDGAR Source Model)

Raw data from SEC Form 4 parsing. Converted to MarketSignal via `from_insider_trade()`.

**Source:** `mae_core/market/apis/sec_edgar/models.py`

| Field | Type | Description |
|-------|------|-------------|
| `filer_name` | string | Insider's name |
| `filer_title` | string | Officer title |
| `filer_relationship` | string | `"Officer"`, `"Director"`, `"10% Owner"` |
| `company_name` | string | Company name |
| `company_cik` | string | SEC CIK number |
| `ticker_symbol` | string | Stock ticker |
| `transaction_date` | string | Trade date |
| `transaction_type` | string | `"A"` (acquired) / `"D"` (disposed) |
| `transaction_code` | string | SEC code: `S`=sale, `P`=purchase, `M`=option, `D`=disposition, `A`=award, `F`=tax, `G`=gift |
| `shares` | float | Share count |
| `price_per_share` | float | Price |
| `total_value` | float | Total dollar value |
| `shares_owned_after` | float | Post-transaction holdings |
| `is_plan_sale` | bool | `True` if 10b5-1 plan detected via footnotes |
| `footnotes` | string | Raw footnote text from filing |
| `filing_date` | string | SEC filing date |
| `accession_number` | string | SEC filing ID |

---

## 11. Signal Flow

```
[Data Source APIs] -> [Raw Source Models] -> [Adapter Functions] -> [MarketSignal]
                                                                        |
                                                    +-------------------+-------------------+
                                                    |                   |                   |
                                            [VelocityDetector]  [FilingTimeAnalyzer]  [JSONL Archive]
                                                    |                   |
                                                    +-------------------+
                                                            |
                                    +---ConvergenceAlerter (global)
                                    +---ConvergenceAlerter (tactical 48h)
                                    +---ConvergenceAlerter (strategic 21d)
                                    +---ConvergenceAlerter (thematic 90d)
                                    +---ClusterDetector (insider trades only)
                                    +---OutcomeCollector -> OutcomeTracker -> ThompsonSampler
                                                            |
                                                    [Intelligence Report]
```
