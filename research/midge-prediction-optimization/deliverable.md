# MIDGE Prediction Optimization — Triadic Deliverable

## Collaborative Output from Lead + Alpha + Beta

**Date:** 2026-02-22
**Scope:** Modifications to maximize MIDGE's pattern detection and financial prediction accuracy

---

## The Verdict

MIDGE has strong architectural DNA (multi-domain convergence, Bayesian learning, fractal self-similarity) but is systematically losing alpha at five layers: **data integrity**, **signal quality**, **architectural gaps**, **timing**, and **execution**. The modifications below are ordered by the triadic consensus on what produces the most value, not by effort.

The overarching principle the triad converged on: **Fix integrity first, unlock architecture second, expand capabilities third.**

---

## Layer 1: Data Integrity (Fix the Foundation)

**Consensus: 3/3 — all analysts agree these must come first**

### 1.1 Fix Outcome Duplication Bug
**What:** The same prediction_id appears 8x in outcomes.jsonl. The Thompson Sampler receives 8 "success" votes from one AAPL prediction, artificially inflating sec_edgar's reliability.
**Where:** `outcome_tracker.py` — add deduplication guard: track evaluated prediction_ids and skip duplicates.
**Why first:** Every future scan compounds this corruption. Cheapest fix with highest long-term leverage.
**Files:** `mae_core/market/intelligence/outcome_tracker.py`

### 1.2 Resolve Contract Signal Ticker Problem
**What:** `from_government_contract()` in signal.py creates signals with `symbol=""`. The OutcomeTracker silently skips predictions without tickers. The entire contract signal class is invisible to the feedback loop.
**Where:** `signal.py:from_government_contract()` and `from_contract_opportunity()` — add a ticker lookup mapping (company name -> ticker from watchlist.json or a dedicated mapping).
**Why first:** Without this, ~20% of signal types can't participate in learning. The outcome collector (Layer 3) would be incomplete from day one.
**Files:** `mae_core/market/signal.py`, `data/midge/watchlist.json` (add company-to-ticker mappings)

### 1.3 Fix Domain Status Table Display Bug
**What:** Scan report accesses `status.get("avg_strength", 0)` but the actual field from `get_domain_status()` is `"strength"`. All domain strengths display as 0.00 in every report.
**Where:** `midge_scan.py:write_report()` — fix the field name.
**Files:** `midge_scan.py`

---

## Layer 2: Signal Quality (Stop the Noise)

**Consensus: 3/3 on items 2.1-2.2; 2/3 on threshold specifics**

### 2.1 Filter 10b5-1 Plan Sales and RSU Vesting
**What:** Scheduled compensation transactions (10b5-1 plans, RSU vesting, option exercises) flood the insider domain with false bearish signals. The 2026-02-22 scan shows Pichai (14 GOOGL signals), Kress (15 NVDA signals) — all scheduled compensation, not informed trading. Transaction codes "D" (delivery/disposition), "A" (award/grant) should be filtered. True 10b5-1 plan transactions can be detected via the Form 4 XML `planName` field.
**Where:** `signal.py:from_insider_trade()` — reduce strength by 75% and confidence to 0.40 for suspected plan sales. Filter transaction codes "D" and "A" from bullish/bearish classification.
**Impact:** Eliminates ~60% of false bearish signals from the insider domain.
**Files:** `mae_core/market/signal.py`, `mae_core/market/apis/sec_edgar/client.py` (parse planName from XML)

### 2.2 Congressional Minimum Trade Size: $50,000
**What:** The 2026-02-22 scan has 85 of 100 congressional signals from $8K trades (Gilbert Cisneros portfolio rebalancing). These have zero informational content and strength 0.02. Transaction costs alone make them unactionable.
**Where:** `midge_scan.py:convert_to_signals()` — skip congressional signals below $50K `amount_high`.
**Why $50K:** At 0.10-0.30% round-trip trading costs on mid-cap stocks, signals below this threshold can't produce economically meaningful alpha even under ideal conditions (Beta's transaction cost analysis).
**Impact:** Eliminates ~85% of congressional noise. Transforms the "congress" domain from noise-dominated to signal-dominated.
**Files:** `midge_scan.py`

### 2.3 Multiple Comparisons Correction in CorrelationTracker
**What:** With N signal pairs, the anomaly threshold of 2.5 sigma produces ~1.26 false positives per cycle at 105 pairs. No Bonferroni or BH correction applied.
**Where:** `correlation_tracker.py` — apply Benjamini-Hochberg correction: `adjusted_threshold = base_threshold * rank / n_pairs`.
**Files:** `mae_core/market/intelligence/correlation_tracker.py`

---

## Layer 3: Architecture (Unlock the Engine)

**Consensus: 3/3 on per-ticker convergence; 2/3 on multi-timeframe**

### 3.1 Per-Ticker Convergence Analysis (Critical)
**What:** The convergence alerter groups signals by domain globally. RTX with insider selling + hiring bullish + contract signals are diluted into global domain averages. GD with mixed insider signals + hiring confirmation is invisible. The system needs a second convergence pass at the symbol level.
**Where:** `convergence_alerter.py` — add `check_ticker_convergence(min_domains=2)` that groups signals by symbol before running convergence.
**Why critical:** This is the architectural bottleneck between "domain-level opinion aggregator" and "ticker-level alpha generator." It's the difference between "the insider domain is bearish" (useless) and "RTX has insider selling + hiring bullish across 3 domains" (actionable).
**Files:** `mae_core/market/intelligence/convergence_alerter.py`, `midge_scan.py` (add ticker convergence to analysis phase)

### 3.2 Wire VelocityDetector into Scan Pipeline
**What:** VelocityDetector exists and works but is not instantiated in `midge_scan.py`. All signal velocities are 0.0. Urgency classification carries no information.
**Where:** `midge_scan.py` — instantiate VelocityDetector, call `record()` for each signal before feeding to alerter, populate `MarketSignal.velocity`.
**Files:** `midge_scan.py`

### 3.3 Wire Filing Time Analyzer into Scan Pipeline
**What:** Filing time behavioral modifiers (Friday dump pattern: -15% confidence, after-hours: -8%) are not flowing into the scan. These are academically grounded signal modifiers going unused.
**Where:** `midge_scan.py` — run FilingTimeAnalyzer on Form 4 and 8-K signals, apply confidence modifiers before feeding to alerter.
**Files:** `midge_scan.py`

### 3.4 Close the Outcome Feedback Loop
**What:** Build `outcome_collector.py` — reads signals from JSONL archives, fetches price at signal time + current price, computes directional accuracy, updates Thompson distributions.
**Where:** New file `mae_core/market/intelligence/outcome_collector.py`.
**Key parameters:**
- Outcome window: 45 days for insider trades, 60 days for clusters, 14 days for congressional (from disclosure), 5 days for 8-K material events, 90 days for contract predictions
- Success threshold: 5% price move in predicted direction (not 2% — the current threshold barely exceeds random baseline)
- Deduplication: Track evaluated prediction_ids, never re-evaluate
**Files:** `mae_core/market/intelligence/outcome_collector.py`, `midge_scan.py` (integrate as optional phase)

### 3.5 Multi-Timeframe Convergence Architecture
**What:** Three independent convergence analyzers with different windows:
- **Tier 1 Tactical (48h):** SEC Form 4, Form 8-K material events
- **Tier 2 Strategic (21d):** Congressional trades, contract awards, insider clusters, politician correlations
- **Tier 3 Thematic (90d):** SAM.gov opportunities, hiring blitzes, contract predictions
**Where:** `midge_scan.py` — instantiate 3 ConvergenceAlerter objects with different window_hours and feed appropriate signal types to each.
**Design note (Alpha's amendment):** Cross-tier convergence (same ticker appearing in multiple tiers) should receive a REDUCED confidence boost vs within-tier convergence, because signals at different timeframes responding to the same event are not independent evidence.
**Files:** `midge_scan.py`, potentially `convergence_alerter.py` (add tier field to alerts)

---

## Layer 4: Timing and Calibration

**Consensus: 2/3 (Beta primary, Lead supportive, Alpha neutral)**

### 4.1 Decay Rate Corrections
All current decay rates vs empirically justified values (treat as priors, not constants):

| Signal | Current | Recommended | Half-Life | Source |
|--------|---------|-------------|-----------|--------|
| InsiderTrade | 0.05 | 0.035 | 20 days | Lakonishok & Lee 2001 |
| ClusterSignal | 0.05 | 0.025 | 28 days | Alldredge 2019 |
| Congressional (from disclosure) | 0.03 | 0.05 | 14 days | PMC 2022, disclosure lag |
| Form 8-K material | 0.03 | 0.25 | 3 days | Market prices binary events in hours |
| Form 8-K informational | 0.03 | 0.05 | 14 days | Moderate persistence |
| ContractPrediction | 0.03 | 0.018 | 39 days | Pre-announcement thesis |
| GovernmentContract (award) | 0.03 | 0.07 | 10 days | Post-announcement drift |
| HiringSignal (pre-announcement) | 0.07 | 0.015 | 46 days | 60-120 day lead time |
| SAM.gov Opportunity | 0.04 | 0.008 | 87 days | Competition periods last months |

**Where:** Each model's `decay_rate` attribute in `signal.py` and respective model files.
**Files:** `mae_core/market/signal.py`, `mae_core/market/apis/sec_edgar/models.py`, `mae_core/market/apis/house_stock_watcher.py`

### 4.2 Scan Frequency Tiering
**When MIDGE moves to scheduled scans:**
- Tier 1 (every 2-4h): SEC EDGAR RSS polling for Form 4 and 8-K
- Tier 2 (daily, post-market): Congressional trades, contract awards, hiring
- Tier 3 (weekly): SAM.gov opportunities

*Not implemented now — manual scan is appropriate until the feedback loop is producing data. But the architecture should anticipate tiered scheduling.*

### 4.3 Insider Trade Strength: Log-Linear Scale
**What:** Replace `min(1.0, value / 1_000_000)` with `min(1.0, log1p(value / 100_000) / log1p(10))`. This preserves differentiation at high values ($1M buy and $10M buy are currently both 1.0).
**Where:** `signal.py:from_insider_trade()`
**Files:** `mae_core/market/signal.py`

---

## Layer 5: Expansion (New Capabilities)

**Consensus: Lead primary, Beta supportive, Alpha cautious ("fix first")**

*These come AFTER Layers 1-4 are complete and the outcome collector has produced 50+ calibrated results.*

### 5.1 Options Flow via Unusual Whales
- $35/month retail tier API
- Domain: "options" in convergence alerter
- Filter: sweep orders > $100K premium, OI change > 500%
- Must go into Tier 1 (Tactical) convergence window — options alpha decays in hours
- Key convergence: options flow + insider buying on same ticker within 72h
**Files:** New `mae_core/market/apis/unusual_whales.py`, update `signal.py`

### 5.2 Senate Stock Watcher
- Mirror `house_stock_watcher.py` for senatestockwatcher.com
- Senate Armed Services Committee has stronger alpha signal than House equivalent
- Same $50K minimum threshold
**Files:** New `mae_core/market/apis/senate_stock_watcher.py`

### 5.3 8-K Text Sentiment via Ollama
- Fetch 8-K text from EDGAR, classify via local Ollama model
- Upgrades "events" domain from rule-based item codes to semantic analysis
- Domain: "events" (existing)
**Files:** New `mae_core/market/edge/form8k_sentiment.py`

### 5.4 Thompson-Weighted Convergence
- Pass ThompsonSampler into ConvergenceAlerter
- Weight each signal's contribution by Thompson-sampled reliability
- **Prerequisite:** Thompson distributions must have 50+ real observations
**Files:** `mae_core/market/intelligence/convergence_alerter.py`

### 5.5 Lag-Correlation Analysis
- Add `compute_lagged_correlation()` to CorrelationTracker
- Discover which signals genuinely lead others (insider buys -> price moves, hiring -> contract awards)
- Requires multi-month signal history in Qdrant
**Files:** `mae_core/market/intelligence/correlation_tracker.py`

### 5.6 Compressed Cluster Detector
- Score time-spread of trades within insider clusters
- All trades within 48h = compression 1.0, +0.10 confidence boost
- Spread over 30 days = compression 0.0, no boost
**Files:** `mae_core/market/edge/cluster_detector.py`

### 5.7 Position Sizing (Simple Rules-Based)
- Base allocation: 5% per convergence alert
- Scale by domain count: 2 domains = 0.5x, 3 = 1.0x, 4+ = 1.5x
- Hard cap: 15% per position
- Upgrade to fractional Kelly after 100+ calibrated outcomes
**Files:** New field on ConvergenceAlert or TradeSignal

---

## What NOT to Build (Triad Consensus)

| Proposed | Decision | Why |
|----------|----------|-----|
| Credit card / transaction data | Skip | $500-2000/month, marginal edge at retail scale |
| Patent filing tracker | Skip | 12-24 month horizon too long for current system |
| Cross-company insider network graph | Defer | Requires substantial graph infrastructure; nice-to-have after core is solid |
| Full Kelly criterion | Defer | Requires calibrated probabilities that don't exist yet |
| Dark pool data | Defer | Confirmation signal only; add after options flow proves value |
| Real-time options flow | Defer | Requires streaming infrastructure MIDGE doesn't have; batch polling is sufficient for now |
| Prospective committee-to-award prediction | Build with compliance flag | Alpha's regulatory concern is valid; signal should carry a visible warning |

---

## Implementation Sequence

```
Phase A: Data Integrity (Layer 1)
  1.1 Fix outcome dedup
  1.2 Resolve contract symbol
  1.3 Fix domain status display

Phase B: Signal Quality (Layer 2)
  2.1 10b5-1 / RSU filter
  2.2 Congressional $50K min
  2.3 Bonferroni correction

Phase C: Core Architecture (Layer 3, items 3.1-3.3)
  3.1 Per-ticker convergence
  3.2 Wire VelocityDetector
  3.3 Wire FilingTimeAnalyzer

Phase D: Feedback Loop (Layer 3, items 3.4-3.5)
  3.4 Outcome collector
  3.5 Multi-timeframe convergence

Phase E: Calibration (Layer 4)
  4.1 Decay rate corrections
  4.3 Log-linear strength scale

Phase F: Expansion (Layer 5, after 50+ outcomes)
  5.1-5.7 as prioritized above
```

**Phases A-C can each be completed in a single session.** Phase D is a full session. Phase E is quick parameter updates. Phase F is ongoing expansion.

---

## Dissenting Notes

### Alpha's Standing Dissent: Confidence Numbers Are Deeper Than Acknowledged

The collaborative deliverable treats confidence calibration as something that gets fixed "eventually" by the outcome collector. I want to surface: the additive confidence formula in the convergence alerter (`0.5 + 0.1 * categories + 0.1 * strength`) is structurally wrong, not just uncalibrated. Two 70% signals combined additively produce 0.90, but the joint probability of both being correct is 0.49 under independence (lower under correlation). This formula should be replaced with a multiplicative or Bayesian combination, not just calibrated by outcome data. The deliverable's implementation sequence defers this to Phase F (Thompson-weighted convergence, item 5.4). I believe the additive formula should be replaced in Phase C alongside per-ticker convergence, because every convergence alert generated between now and Phase F will carry inflated confidence.

**The team's response:** Acknowledged. The additive formula IS wrong. But replacing it with a proper Bayesian combination requires knowing the prior probability of each domain's accuracy — which requires outcome data we don't have yet. The additive formula is a known-bad placeholder. Replacing it with a different formula that's also uncalibrated doesn't improve accuracy, just changes the shape of the error. We accept the dissent and note that Phase F (item 5.4) should be prioritized as soon as 50+ outcomes exist.

---

*Endorsed by: Lead, Alpha, Beta*
*Date: 2026-02-22*
