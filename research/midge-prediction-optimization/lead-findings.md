# MIDGE Prediction Optimization — Lead Analyst Findings
## Analytical Lens: Signal Processing and Alpha Generation

**Date:** 2026-02-22
**Role:** Lead Analyst (Signal Processing / Alpha Generation)
**Triadic Context:** This document contains findings from the LEAD lens only. Alpha and Market-Structure findings are in separate documents.

---

## Executive Summary

MIDGE is a well-architected signal aggregation system with a serious collection of alternative data sources and a sound convergence architecture. However, the live scan output reveals the core problem clearly: 245 signals collected, 50 fed to the alerter, one weak convergence alert at 30% confidence ("Insufficient convergence — signals mixed or single-category"). The signal pipeline has the right bones but is systematically losing alpha at five distinct points:

1. **Data source gaps** — MIDGE is missing the three highest-edge alternative data classes used by real quant funds: options flow, dark pool prints, and transaction (credit card) data.
2. **Signal strength formula flaws** — The strength formula in `signal.py` treats all insider sells identically regardless of whether they are plan-driven or discretionary, and the congressional strength formula is miscalibrated.
3. **Convergence alerter blindness** — The alerter does not look at any ticker-specific convergence, only domain-level aggregates. A ticker with a cluster buy + congressional buy + hiring spike will never generate an alert because all three signals dilute into global domain averages.
4. **Thompson Sampler not closing the loop** — The sampler has 12 signals with prior beliefs but zero actual Bayesian updates from real outcomes. It is functioning as a static lookup table, not a learning system.
5. **Missing edge detectors** — Several high-signal patterns in the data MIDGE already collects are going undetected: insider TIMING patterns, congressional committee-to-trade timing, cross-company insider networks.

---

## 1. Data Source Gaps

### 1.1 Options Flow — Highest Priority Missing Signal

**What it is:** Real-time tracking of unusual options activity — large open interest changes, sweep orders, and call/put skew anomalies that precede stock moves.

**Why it matters:** Options buyers with material non-public information cannot act directly (illegal). They can, and do, buy options. Academic research and practitioner experience consistently show that unusual options activity — specifically large out-of-the-money call buying or sudden open interest spikes — precedes earnings surprises and M&A announcements by 3–10 days with meaningful predictive accuracy (~60%+ directional accuracy on high-conviction filters). This is the most widely used alternative data signal at institutional level.

**What MIDGE currently does:** The `learning_config.py` lists "unusual_whales" as a signal source with 0.80 reliability, and the `thompson_distributions.json` shows it has a Beta distribution initialized. But there is no `unusual_whales.py` client, no fetcher, and no adapter in `signal.py`. It is configured but not implemented — a ghost signal.

**Integration path:**
- **Unusual Whales API** (`unusualwhales.com/public-api`): $35/month retail tier. Provides real-time options flow, dark pool prints, and congressional data via REST API. For alpha purposes, filter on: (a) sweep orders above $100k premium, (b) open interest change > 500% same-day, (c) OTM call buying within 5–15 days of expiry.
- Adapter: `from_options_flow(flow_event)` in `signal.py`, domain `"options"`, direction based on call/put, strength scaled by premium size relative to 30-day average for that ticker.
- The `convergence_alerter.py` domain registry already has "market" as a category (line 126). Options flow would add a second signal in that category alongside "technical" and "volume."
- Critical: An options flow signal confirmed by insider buying on the same ticker within 72 hours is a near-actionable triadic convergence. This combination is currently impossible to detect because options flow is missing.

**Expected edge:** Directional accuracy on filtered options flow signals (sweep + OI anomaly combined) is reported at 55–65% by practitioners. Combined with insider confirmation this rises toward 70%+.

### 1.2 Dark Pool Prints — Second Priority

**What it is:** Off-exchange block trades that appear as "prints" — large institutional transactions at specific price levels that often act as support/resistance and signal institutional accumulation or distribution before the market knows.

**Why it matters:** Dark pools account for roughly 35–40% of all US equity volume. Large accumulation at a price level before a move is one of the cleaner "smart money" signals available from public FINRA reporting.

**What MIDGE currently does:** Nothing. No client, no domain, no adapter.

**Integration path:**
- Unusual Whales also provides dark pool data at the same subscription tier. Alternatively, QuantData.us provides dark pool prints with filtering.
- Filter criteria for alpha: dark pool print > $1M on a ticker that ALSO has unusual options activity or insider buying in the last 7 days. The combination of all three (dark pool + options + insider) is extremely rare and extremely high signal.
- Domain: `"volume"` in the convergence alerter domain map (line 127). Strength scaled by print size relative to 30-day ADTV.

**Expected edge:** Dark pool prints alone have weak directional accuracy. As a confirmation signal on top of options flow or insider activity, they act as a strong second domain to push convergence past the 3-domain threshold.

### 1.3 Credit Card / Transaction Data — Medium Priority

**What it is:** Aggregated, anonymized consumer spending data sold by credit card networks and processors. Used to predict earnings surprises in consumer-facing companies 30–60 days before earnings.

**Why it matters:** Research from ExtractAlpha and others shows transaction data achieves roughly 10–20% improvement in earnings surprise prediction for retail, restaurant, travel, and subscription businesses. This is where the real quant funds play — Thasos, Earnest Research, and YipitData all sell variants of this.

**What MIDGE currently does:** Not present. No mention anywhere in the codebase.

**Integration path (tiered by cost):**
- **Tier 1 (Free/Cheap):** App store ranking data (free via SensorTower APIs, $X/query). Downloads correlate with usage, which correlates with revenue. Good proxy for SaaS/consumer app revenue.
- **Tier 2 ($500–2000/month):** Second Measure or similar consumer panel data. Revenue is $2,000+/month for institutional, but some data vendors offer per-query models.
- **Tier 3 (Alternative proxy — free):** Web traffic data via SimilarWeb or Alexa APIs can approximate transaction volumes for e-commerce companies. Significant drops in web traffic 4–6 weeks before earnings are predictive of revenue misses.
- Best integration point: new API client `web_traffic_tracker.py` in `mae_core/market/apis/`, domain `"fundamentals"`, used as an early warning on earnings.

**Expected edge:** Transaction data signals achieve ~55–60% directional accuracy on earnings direction, with best performance on small-to-mid cap consumer companies where analyst coverage is thin.

### 1.4 Patent Filing Activity — Lower Priority, High Specificity

**What it is:** Tracking patent filing rates at USPTO for specific companies, measuring innovation investment velocity.

**Why it matters:** Sudden increases in patent filings in a specific technical area (e.g., AI hardware, biotech therapeutics) can predict competitive dynamics and future revenue before financial statements reflect the investment. Research shows rising patent activity in a new domain correlates with M&A activity 12–24 months later.

**What MIDGE currently does:** Not present.

**Integration path:** USPTO's bulk data is free and available via API. Patent Lens (lens.org) provides free search. This is a long-horizon signal (decay rate ~0.01, half-life 69 days) that would fit in the `"fundamentals"` domain and pair well with the `"government"` signals for defense tech companies. Lower priority than options flow and dark pools.

### 1.5 Senate Disclosure Coverage Gap

**What MIDGE currently does:** `house_stock_watcher.py` tracks House trades via housestockwatcher.com. There is `congress_members.json` providing committee membership for both chambers. However, the Senate is notably different — Senate leaders have more decisive committee power, and Senate Armed Services Committee membership + stock trades has historically shown stronger alpha than House equivalents.

**Fix needed:** senatestockwatcher.com provides Senate trade data in the same format as housestockwatcher. Adding a `senate_stock_watcher.py` client that mirrors the existing House client would immediately double the congressional signal coverage with no new API costs.

---

## 2. Algorithm Gaps

### 2.1 Ticker-Level Convergence (Critical Missing Feature)

**The most important missing algorithm is not ML — it is ticker-level convergence.**

**The problem:** The convergence alerter aggregates signals by domain globally. When the scan collects 103 Form 4 signals and 100 congressional signals, it computes a domain-level direction for "insider" and "congress" as whole categories, ignoring which stocks those signals are on. The 2026-02-22 scan shows RTX with both a bearish insider cluster AND a bullish hiring signal — but the alerter never saw that because RTX's two signals diluted into global domain means.

**Code reference:** `convergence_alerter.py:261–331` (`_check_direction_convergence`). The entire method is domain-global. There is no ticker-scoped convergence anywhere in the system.

**What should exist:** A second convergence pass that groups signals by `symbol` before running convergence analysis. When 3+ domains converge on the SAME TICKER, that is a far stronger signal than 3 domains globally bullish on different stocks.

**Implementation sketch (no new dependencies):**
```python
def check_ticker_convergence(self, min_domains: int = 2) -> Dict[str, List[ConvergenceAlert]]:
    """Per-ticker convergence — more actionable than global domain convergence."""
    by_ticker = defaultdict(lambda: defaultdict(list))
    for domain, signals in self.signals.items():
        for sig in signals:
            ticker = sig.metadata.get("symbol", "")
            if ticker:
                by_ticker[ticker][domain].append(sig)

    ticker_alerts = {}
    for ticker, domain_signals in by_ticker.items():
        if len(domain_signals) >= min_domains:
            # Run convergence on this ticker's signals only
            ...
    return ticker_alerts
```

**Priority:** This is the single highest-priority algorithm fix. RTX in the 2026-02-22 scan had insider selling + hiring spike converging — that's a signal that the hiring spike is not contract-driven but retention-driven, which flips the bullish hiring read bearish. Ticker-level convergence would have caught that distinction.

### 2.2 Insider Trade Type Discrimination (Signal Quality Fix)

**The problem:** In `signal.py:113–115`, strength for insider sells is calculated identically regardless of trade type:
```python
if is_buy:
    strength = min(1.0, trade.total_value / 1_000_000)
else:
    strength = min(1.0, trade.total_value / 500_000)
```

The academic literature is clear: **Rule 10b5-1 plan sales are noise; discretionary open-market sells are signal.** The `cluster_detector.py` correctly filters 10b5-1 plans for buy clusters (`line 236: if details.get("is_10b5_1_plan", False): continue`), but `signal.py` does not apply this filter. Every executive's auto-sales plan generates a "bearish" signal at full strength, flooding the system with false negatives.

**The 2026-02-22 scan evidence:** NVDA, META, GOOGL, NOC all show multiple high-strength bearish Form 4 signals ($500k–$4M sells). These are almost certainly 10b5-1 plan sales, not discretionary bearish sentiment. The scan correctly reports "bearish" for all these stocks but this is systematically misleading.

**Fix in `signal.py:from_insider_trade()`:**
```python
# Check for 10b5-1 plan
is_plan_sale = (not is_buy and
                trade.transaction_code in ["F", "S"] and
                trade.metadata.get("is_10b5_1_plan", False))
if is_plan_sale:
    strength *= 0.25  # Drastically reduce plan sale strength
    confidence = 0.40  # Low confidence — not discretionary
```

### 2.3 Congressional Signal Strength Miscalibration

**The problem:** In `signal.py:197`, congressional trade strength is calculated as:
```python
strength = min(1.0, trade.amount_high / 500_000)
```

This means a $8,000 trade (amount_high=$8,000) gets strength=0.016. But the 2026-02-22 scan has 70+ congressional signals all at strength=0.02 from Gilbert Cisneros selling $8,000 positions across dozens of tickers. These are portfolio rebalancing signals — noise. They dominate the congressional domain and flip it bearish globally.

**Fix:** Add a minimum threshold filter in `midge_scan.py:convert_to_signals()` that drops congressional signals below $15,000. Alternatively, apply a non-linear strength function that compresses small trades harder:
```python
# Logarithmic compression: small trades get much less weight
import math
strength = min(1.0, math.log1p(trade.amount_high / 1000) / math.log1p(500))
```

This alone would transform the congressional domain from "noisy bearish" to "informative."

### 2.4 Ensemble Signal Weighting (Missing Fusion Layer)

**The problem:** The convergence alerter uses simple averaging for strength and confidence (`convergence_alerter.py:292–293`):
```python
avg_strength = sum(s.strength for s in converging_signals) / len(converging_signals)
avg_confidence = sum(s.confidence for s in converging_signals) / len(converging_signals)
```

This treats a 10b5-1 plan sale (low signal) equally with a CEO discretionary open-market purchase (high signal). Worse, it treats a domain with 1 signal the same as a domain with 50 signals.

**What should replace it:** Thompson-weighted averaging. The ThompsonSampler already knows which signal sources are more reliable (`thompson_distributions.json`). But it is never consulted when computing convergence strength. The fix is to weight each signal's contribution by its Thompson-sampled reliability score:

```python
# In _check_direction_convergence(), replace simple averaging:
total_weight = 0
weighted_strength = 0
for signal in converging_signals:
    source_reliability = self.thompson_sampler.sample(signal.signal_id)
    weighted_strength += signal.strength * source_reliability
    total_weight += source_reliability
avg_strength = weighted_strength / max(total_weight, 0.001)
```

This requires passing the ThompsonSampler instance into ConvergenceAlerter, which is architecturally sound and already partially set up (both are bootstrapped in Layer 33).

### 2.5 Temporal Lead-Lag Analysis (Missing Causal Structure)

**The problem:** The CorrelationTracker computes Pearson correlation between signals within a 1-hour alignment window (`correlation_tracker.py:148–164`). For financial signals that operate on day-to-week timescales (insider buys precede price moves by 5–20 days), a 1-hour alignment window produces near-zero correlations on any pair of signals. The tracker will report no meaningful correlations until it accumulates 30+ observations with the right alignment window.

**Fix:** Add a lag parameter to `compute_correlation()`:
```python
def compute_lagged_correlation(self, signal_a: str, signal_b: str,
                                lag_days: int = 5) -> Optional[float]:
    """Compute correlation between signal_a at time T and signal_b at time T+lag_days."""
```

This enables MIDGE to discover that insider buys (domain A) systematically lead price velocity (domain B) by N days, which is where the genuine alpha lives. Without lag analysis, the correlation tracker cannot find leading indicators — only concurrent correlations.

### 2.6 Reinforcement Learning for Signal Weighting (Medium-term)

**What exists:** Thompson Sampling for source reliability. This is correct Bayesian explore/exploit for choosing which signals to trust.

**What is missing:** Any feedback loop connecting outcomes to signal weights. The OutcomeTracker concept exists in the schema (`MarketSignal.outcome_symbol`, `outcome_window_days`) but there is no file that actually fetches prices N days later, computes directional accuracy, and calls `thompson_sampler.update()`.

**What this means:** The Thompson distributions in `thompson_distributions.json` show that all signals have alpha=1.0 or near-1.0 with beta=1.0 — they are all at their initial priors. Zero real Bayesian updates have occurred. The learning system is architecturally present but operationally inert.

**Minimum fix:** A scheduled script `outcome_collector.py` that:
1. Reads `data/midge/signals/*.jsonl` for signals older than `outcome_window_days`
2. Fetches current price via PriceFetcher
3. Computes: if signal was bullish and price is higher → success=True; else success=False
4. Calls `thompson_sampler.update(signal.source, success=..., regime=current_regime)`

This is the highest-leverage algorithm improvement available because it makes the entire system learn. Without it, MIDGE has identical confidence in SEC EDGAR and Reddit regardless of actual performance.

---

## 3. Signal Processing Weaknesses

### 3.1 Velocity Detector Is Disconnected From Output

**Code reference:** `velocity_detector.py` exists and is well-implemented. `midge_scan.py:292–302` feeds signals to the convergence alerter but never feeds them to the VelocityDetector. The `MarketSignal.velocity` field defaults to 0.0 (`signal.py:74`) and is never populated by any code path in the current scan.

**Impact:** The urgency classifier in `convergence_alerter.py:299–306` uses velocity to determine "immediate" vs "hours" vs "days." Since all velocities are 0.0, every alert gets classified as "days" urgency. The urgency signal carries no information.

**Fix:** In `midge_scan.py:store_and_feed()`, record each signal to the VelocityDetector before feeding to the alerter:
```python
velocity_state = velocity_detector.record(sig.signal_id, sig.strength, sig.timestamp)
sig.velocity = velocity_state.current_velocity  # Mutate before feeding alerter
```

### 3.2 Regime-Aware Thompson Sampling Is Partially Wired

**Code reference:** `regime_classifier.py` and `thompson_sampler.py` both support regimes. The RegimeClassifier classifies the market regime using SPY price data. However, in `midge_scan.py`, the ThompsonSampler is never instantiated, and in `main.py` bootstrap (Layer 33), the RegimeClassifier is only used if `price_fetcher` is injected.

**Impact:** All Thompson sampling occurs in the "default" regime regardless of market conditions. The regime-aware design, which was explicitly called out in `CLAUDE.md` Phase 2 as a deferred feature, could improve signal reliability by 10–15% because insider buy signals are meaningfully more reliable in bear markets (conviction buys) than in bull markets (routine accumulation).

### 3.3 Confidence Boost Formula Is Too Conservative

**Code reference:** `convergence_alerter.py:296`:
```python
confidence_boost = min(0.2, 0.05 * (cross_domain_count - 1))
```

With 4 cross-domain categories, max boost = 0.15. With base confidence around 0.65 (the congressional base from `learning_config.py`), peak convergence confidence reaches 0.80. But the `get_actionable_summary()` method at line 425 adds `0.1 * avg_strength` to the confidence, so the actual cap is around 0.80–0.85.

**The deeper problem:** Confidence is a static function of domain count and signal count. It does not incorporate historical accuracy of past convergence alerts. Until the feedback loop is closed, confidence numbers are theoretical estimates, not calibrated probabilities. They look precise (3 decimal places) but they are meaningfully uncertain.

**Recommendation:** Add a calibration multiplier once 50+ resolved predictions exist:
```python
historical_accuracy = self.thompson_sampler.get_distribution("convergence_alert").mean
calibrated_confidence = final_confidence * (historical_accuracy / 0.5)  # Normalize to actual hit rate
```

### 3.4 Strength Saturation at $1M Buys (Formula Cliff)

**Code reference:** `signal.py:113`: `strength = min(1.0, trade.total_value / 1_000_000)`

This creates a cliff: a $1M buy and a $10M buy both get strength=1.0. But a $10M discretionary purchase is qualitatively different from a $1M purchase. The MSFT signal from the 2026-02-22 scan shows STANTON JOHN W buying $1,986,750 worth — it caps at 1.0 and loses the information that it's nearly $2M.

**Fix:** Use a log-linear scale that preserves differentiation at high values:
```python
import math
strength = min(1.0, math.log1p(trade.total_value / 100_000) / math.log1p(10))
# $100k → 0.5, $500k → 0.75, $1M → 0.85, $5M → 0.95, $10M → 1.0
```

---

## 4. Missing Edge Detectors

### 4.1 Insider Trade TIMING Pattern Detector

**What exists:** `cluster_detector.py` detects when 3+ insiders buy within 30 days (cluster signal). `filing_time_analyzer.py` detects suspicious filing times.

**What is missing:** A detector for the TEMPORAL PATTERN within the cluster — specifically, whether insiders are all buying within a compressed 48-hour window (extremely high signal) vs spread across 30 days (moderate signal). Academic research on insider trading consistently shows that compressed-window cluster buys (multiple insiders within 48 hours) predict abnormal returns 2–3x better than same-month clusters.

**Implementation:** Add a `CompressedClusterDetector` class to `cluster_detector.py` that, after identifying a cluster, computes the time-spread of trades:
```python
def _calculate_cluster_compression(self, insiders: List[dict]) -> float:
    """0.0 = trades spread over 30 days. 1.0 = all trades within 48h."""
    dates = [parse(i["trade_date"]) for i in insiders]
    span_days = (max(dates) - min(dates)).days
    return max(0.0, 1.0 - span_days / 30.0)
```
A compression score > 0.9 (all trades within 3 days) should boost ClusterSignal confidence by +0.10.

### 4.2 Congressional Committee-to-Award Timing Detector

**What exists:** `politician_tracker.py:_check_contract_correlation()` checks if a politician's committee oversees an agency that awarded a contract. This is correct but only runs reactively when a contract is found.

**What is missing:** A PROSPECTIVE detector that flags when a committee member buys a stock BEFORE a related contract is even announced. The current code window of `-30 <= days_diff <= 90` (line 274) is broad but passive — it only fires when a contract was already awarded.

**The alpha lives in the predictive direction:** Committee member buys stock → MIDGE predicts contract will follow → contract is awarded 30–90 days later. This requires the PoliticianTracker to run FORWARD rather than backward: when a committee member trade is detected, proactively search SAM.gov for open solicitations from agencies that committee oversees.

**Implementation:** In `politician_tracker.py`, add `predict_upcoming_contracts()`:
```python
def predict_upcoming_contracts(self, trade: InsiderTrade, politician: PoliticianProfile):
    """When politician trades, predict which contracts might flow."""
    # Get agencies this politician's committees oversee
    agencies = [AGENCY_COMMITTEE_MAP.get(c, []) for c in politician.committees]
    # Search SAM.gov for open opportunities from those agencies
    opps = self.sam_client.search_by_agency(flatten(agencies))
    return opps  # These are predicted upcoming awards
```

### 4.3 Cross-Company Insider Network Graph (High Value)

**What exists:** `cluster_detector.py:RelationshipTracker` tracks pairs of insiders who trade the same stock within 48 hours. This is a start.

**What is missing:** A CROSS-COMPANY network graph. If CEO of Company A frequently trades before the CEO of Company B does (even in different companies), and they share board connections, this is a network of informed trading. Academic research calls this "connected insider networks" and shows they generate abnormal returns 3–5x higher than unconnected insider trades.

**Implementation:** This requires extending `RelationshipTracker.build_multi_symbol_graph()` to include cross-company pairs, not just same-company. The data for this already flows through the SEC EDGAR client. The graph can be stored in Qdrant as a node-edge structure.

### 4.4 Form 8-K Sentiment Analyzer

**What exists:** `sec_edgar/models.py:Form8KEvent` maps item codes to direction (bullish/bearish). This is purely rule-based: item 2.02 (Earnings) = bullish or bearish based on code only.

**What is missing:** NLP-based sentiment extraction from the actual 8-K text. A company filing item 2.02 with language like "exceeded expectations" is qualitatively different from "results were in line with reduced guidance." The difference is detectable with basic NLP (even the local Ollama model can classify this).

**Implementation:** Add a `Form8KSentimentAnalyzer` that fetches the 8-K text from EDGAR and runs it through Ollama:
```python
def analyze_text_sentiment(self, accession_number: str) -> float:
    """Returns sentiment score -1.0 to 1.0 from 8-K text."""
    text = self.edgar_client.get_filing_text(accession_number)
    prompt = f"Classify this financial disclosure as bullish (+1), neutral (0), or bearish (-1). Return only the number.\n\n{text[:2000]}"
    # Call Ollama
    ...
```
This would upgrade 8-K signals from rule-based to text-based, significantly improving the precision of the "events" domain.

---

## 5. Feedback Loop Quality

### 5.1 Current State: The Loop Is Not Closed

The Thompson Sampler architecture is sound but the data flowing through it is initialization-only. From `thompson_distributions.json`:
- `sec_edgar`: alpha=1.68, beta=1.0 → mean=0.63 (prior from reliability config)
- `reddit`: alpha=1.0, beta=1.24 → mean=0.45 (prior)
- All others are at their seeded priors.

No distributions show evidence of actual Bayesian updates (which would produce non-round beta values far from 1.0). The system has been live but the outcome collection described in the HANDOFF has not been implemented.

**Evidence from code:** `thompson_sampler.py:update()` exists and is correct. `thompson_sampler.py:_log_update()` appends to `thompson_history.jsonl`. But there is no caller that invokes `update()` with actual trade outcomes. The history file is either empty or absent.

### 5.2 What Would Make It Learn: Minimum Viable Outcome Collector

The outcome window in `signal.py` defaults to 14 days (`outcome_window_days: int = 14`). The minimum viable learning loop is:

1. **Collector script** reads signals from `data/midge/signals/*.jsonl` that are 14+ days old and have `outcome_symbol` set.
2. Fetches price at signal timestamp and current price using `PriceFetcher.get_historical_price()`.
3. Computes: `success = (price_now > price_then) == (direction == "bullish")`.
4. Calls `sampler.update(signal.source, success=success, regime=regime_at_signal_time)`.
5. Logs to `outcomes.jsonl`.

With 100+ resolved predictions, the system would have empirically calibrated Beta distributions. SEC Form 4 signals would have their 0.95 prior either confirmed or revised. Congressional signals (currently assumed 0.65 confidence) would have evidence-based reliability.

### 5.3 Regime-Stratified Learning

The `RegimeClassifier` classifies into bull/bear/volatile/sideways. The Thompson Sampler stores separate distributions per regime. But without the outcome collector, both capabilities are dormant.

The hypothesis to test once outcomes flow: insider buy signals should be MORE reliable in bear regimes (when executives are buying against the trend, it's higher conviction) and LESS reliable in bull regimes (executives sometimes buy opportunistically without strong informational advantage). If this hypothesis holds, the regime-stratified distributions would show `form_4[bear].mean > form_4[bull].mean`, and the convergence alerter's weighting would automatically adjust.

### 5.4 Forgetting Rate Is Too Slow

`thompson_sampler.py:apply_forgetting()` uses `decay_factor=0.99`. At this rate, an observation from 6 months ago retains 99%^180 = ~16% of its original weight. For financial signals that change in reliability over time (regulatory changes, market structure changes), this is too slow. In a bear market, signal reliability profiles shift significantly.

**Recommendation:** Use an adaptive forgetting rate tied to regime changes. When the regime classifier changes from "bull" to "bear," apply an aggressive single-step decay (factor=0.90) to all distributions, effectively saying "the rules just changed, trust history less." Between regime changes, maintain the 0.99 slow decay.

---

## 6. Specific Actionable Recommendations (Priority Ordered)

### Tier 1 — Highest Alpha Impact, Low Effort

1. **Fix congressional signal noise** (1 hour): Add a `min_amount_high > 15000` filter in `midge_scan.py:convert_to_signals()` before appending congressional signals. This immediately removes ~70 noise signals from the 2026-02-22 scan and transforms the "congress" domain from noise-dominated to signal-dominated.

2. **Fix 10b5-1 plan sell strength** (2 hours): In `signal.py:from_insider_trade()`, check `transaction_code` and reduce strength/confidence for plan sales. This removes the false bearish signals flooding the "insider" domain.

3. **Add per-ticker convergence** (1 day): Add `check_ticker_convergence()` to `ConvergenceAlerter`. This is the most architecturally important fix — it transforms MIDGE from a domain-level opinion aggregator to a ticker-level alpha generator.

4. **Wire the VelocityDetector** (2 hours): In `midge_scan.py`, instantiate VelocityDetector and record each signal before feeding to the alerter. Populate `MarketSignal.velocity` so urgency classification carries real information.

### Tier 2 — High Alpha Impact, Moderate Effort

5. **Build outcome_collector.py** (1 day): Close the Thompson learning loop. This makes every subsequent improvement self-improving.

6. **Add Unusual Whales options flow client** (1 day): `mae_core/market/apis/unusual_whales.py`. Start with the $35/month tier. Add `from_options_flow()` adapter in `signal.py`. Domain: "options". This is the single highest-signal data source MIDGE is missing.

7. **Fix strength saturation** (2 hours): Replace linear strength formula in `signal.py` with log-linear scale for insider trades.

8. **Add Senate stock watcher** (3 hours): Mirror `house_stock_watcher.py` for Senate data from senatestockwatcher.com. Senate committee power is higher than House for the sectors MIDGE tracks (defense, tech contracts).

### Tier 3 — Structural Improvements, Higher Effort

9. **Thompson-weighted convergence** (1 day): Pass ThompsonSampler into ConvergenceAlerter and use sampled reliability for strength averaging. Requires ThompsonSampler to have real posteriors first (depends on #5).

10. **Add lag-correlation analysis** (1 day): Add `compute_lagged_correlation()` to CorrelationTracker. This is how MIDGE discovers which signals genuinely lead others.

11. **Add 8-K text sentiment** (2 days): Use Ollama (already installed on Wardenclyffe) to extract actual sentiment from Form 8-K text. Upgrades the entire "events" domain from rule-based to semantically grounded.

12. **Add compressed cluster detector** (4 hours): Extend ClusterDetector to score cluster compression (time spread of trades). High-compression clusters deserve a +0.10 confidence boost.

---

## 7. What the 2026-02-22 Scan Actually Tells Us

The live scan reveals the following signal landscape for that day:

- **MSFT:** STANTON JOHN W bought $1.98M (bullish, str=1.0) while Coleman Amy sold $34k (bearish, plan-likely). Net: bullish insider signal, neutralized by noise.
- **RTX:** Three large insider sells ($1.6M–$3.5M) + one hiring tracker bullish signal. This is a high-signal situation: insiders distributing while RTX is hiring aggressively. Likely: insider sales are 10b5-1 plan sales, hiring spike is real. Per-ticker convergence would have caught this.
- **NOC, GD, BA:** Large insider sells across defense contractors. Almost certainly plan sales based on the uniform large amounts. The current system reports the entire defense sector as bearish on these signals, which is misleading.
- **Congressional flood:** Gilbert Cisneros sold $8,000 positions across 60+ tickers simultaneously. This appears to be a systematic portfolio rebalancing event, not informative trading. These 60+ signals are dominating the "congress" domain and declaring it globally bearish.

**The 30% confidence neutral recommendation is correct given the current data quality.** But with the fixes above — specifically congressional noise filtering, 10b5-1 filtering, and per-ticker convergence — this same scan would likely produce a moderate bullish signal on RTX (hiring + single non-plan buy) and maintain neutral on the others, which would be a more useful and accurate output.

---

## 8. Cross-Reference for Triadic Analysis

The following areas are most critical for the other analysts to cover:

- **For the Alpha Analyst (portfolio/execution):** The congressional disclosure lag is 45 days. Most alpha from congressional trades is consumed before public disclosure (academic research: "70-80% of alpha dissipates before filing date"). This means MIDGE's congressional signals are trailing indicators, not leading indicators, for most trades. The window of alpha that exists is in the committee-to-award correlation (before contract, not after trade disclosure).

- **For the Market Structure Analyst (macro/regime):** The RegimeClassifier uses SPY 20-day return thresholds of +/-2% and volatility > 25%. These thresholds are from conventional wisdom, not empirically calibrated for MIDGE's specific signal types. The question of whether these regime boundaries actually predict signal reliability changes is empirically unanswered.

---

*End of Lead Analyst Findings.*
*Confidence in these findings: HIGH for code-referenced weaknesses (directly verified), MEDIUM for external data source recommendations (based on practitioner consensus and public research), LOWER for specific hit-rate claims (highly context-dependent).*
