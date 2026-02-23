# MIDGE Adversarial Analysis — Witness Alpha
## Risk, Calibration, and Adversarial Analysis

**Analyst Lens:** Skeptic — false positives, overfitting, survivorship bias, data snooping, regulatory risk, confidence miscalibration

**Codebase reviewed:** `C:\Users\baenb\projects\MIDGE`
**Date of analysis:** 2026-02-22
**Files read in full:** 14 source files, 6 data files, 5 scan/log files

---

## Executive Verdict

MIDGE is an architecturally sophisticated signal aggregation system built on sound organizational principles. The Mae triadic framework is genuinely novel and the multi-domain convergence concept has theoretical support. However, the predictive layer — the part that actually has to be right to make money — contains multiple compounding weaknesses that would likely result in real-world underperformance or loss. The most dangerous property is not any single flaw but the system's tendency toward confident-sounding output that has no empirical anchor. A 0.75 confidence score is displayed on the dashboard. There is no evidence anywhere in the codebase that 0.75 maps to a 75% hit rate. It does not. The analysis that follows quantifies why.

---

## 1. False Positive Analysis

### 1.1 Insider Cluster Detector

**File:** `mae_core/market/edge/cluster_detector.py`

**The core confidence assignment (lines 375-381):**
```python
confidence = 0.70
confidence += min(0.15, (insider_count - 3) * 0.05)  # Extra insiders
if has_csuite:
    confidence += 0.05
if avg_conviction > 0.4:
    confidence += 0.05
confidence = min(0.95, confidence)
```

This is a fabricated scoring function. The base confidence of 0.70 for a 3-insider cluster is not derived from any backtest, academic citation, or calibration exercise. It is an assertion. The research the docstring cites — "Cluster buys (3+ insiders within 30 days) produce higher returns [high]" — is referenced as coming from "universal_vault_v2 predictions," which is MIDGE's own internal knowledge store, not an independent study.

**What the actual academic literature says:** Jeng, Metrick, and Zeckhauser (2003) found insider purchases earn abnormal returns of roughly 50 basis points per month, averaging to approximately 6% per year before transaction costs and taxes. Lakonishok and Lee (2001) found that insider buying predicts returns, but the effect is concentrated in small-cap stocks and largely disappears in large-caps. A 2026 SSRN paper titled "The Death of Insider Trading Alpha" notes practitioner backtests routinely report performance "far exceeding what is documented" in peer-reviewed literature, suggesting the alpha of publicly observable Form 4 data has been heavily arbitraged.

**False positive modes for cluster detection:**
1. **10b5-1 plan contamination.** The filter at `cluster_detector.py:239` only excludes 10b5-1 plans when `details.get("is_10b5_1_plan", False)`. This field is frequently absent from raw Form 4 data. Transaction code "S" (sale) is not reliably tagged as planned. Code "D" (delivery/disposition) used for option exercises on vesting is correctly excluded, but many scheduled-plan purchases arrive tagged as open-market code "P" with no plan indicator in the payload. Large-cap tech companies routinely have 30-40% of insider "purchases" under pre-established plans. These carry near-zero informational content yet would pass the filter and contribute to cluster counts.

2. **Multiple transactions from one insider counted as cluster members.** The `_aggregate_by_insider` function at line 248 groups by `filer_name`. If NVDA CFO Kress Colette executes 14 option-exercise transactions in one day (as seen in `data/midge/signals/2026-02-22.jsonl`, lines 50-73), these would be collapsed to one insider entry. However, separate insiders with small trades on different days within the 30-day window would each count as a "cluster member" even when their trades are scheduled vests, not independent buy decisions.

3. **Relationship tracker false correlation (lines 519-542).** The `correlation_score` function is `min(0.95, 0.5 + (rel.trades_together * 0.1))`. After just 5 co-trades within 48 hours, two insiders reach 1.0 correlation. At large companies with quarterly vesting schedules, every officer on the same vesting schedule will trigger this — dozens of "correlated" pairs that are simply the payroll calendar.

**Estimated realistic false positive rate for "cluster = bullish signal":** 40-60% of clusters on S&P 500 large-caps will be composed primarily of planned transactions. The cluster detector makes no distinction between independent open-market conviction purchases and scheduled vesting events. The practical true positive rate is likely in the 0.50-0.55 range for large-caps, barely above coin-flip.

---

### 1.2 Contract Predictor

**File:** `mae_core/market/edge/contract_predictor.py`

**The additive confidence model (lines 81-110):**
```python
if self.is_active_bidder:
    breakdown["active_bidder"] = 0.20
if self.hiring_blitz_detected:
    base = 0.15  # to 0.25 based on spike ratio
    breakdown["hiring_blitz"] = base
if self.insider_buying_detected:
    base = 0.10  # to 0.20 based on value
    breakdown["insider_buying"] = base
if self.historical_winner:
    breakdown["historical_pattern"] = 0.15
self.confidence = min(0.95, sum(breakdown.values()))
```

This is simple additive score construction with no probabilistic justification. The component weights (0.20, 0.25, 0.20, 0.15) are not derived from any regression, no base rate is established, and there is no accounting for the prior probability of any given bidder winning a defense contract.

**The base rate problem is severe.** A government contract solicitation may attract 3-15 serious bidders. The base probability that any one company wins is therefore 7-33%. The `is_active_bidder` flag is set `True` by default in `_analyze_bidder` at line 189 (`is_active_bidder=True`), meaning every company in `DEFENSE_CONTRACTORS` starts with a 0.20 confidence boost automatically. This alone pushes the starting confidence above a realistic prior. Even before any other signals fire, MIDGE would assign Lockheed Martin 0.20 confidence to win any defense contract simply because they exist.

**The hiring blitz signal has no contract-specific content.** The job tracker uses the RapidAPI Jobs API, which aggregates all open positions. A defense contractor hiring 3x normal levels in aerospace engineering is: (a) responding to any new contract win (not predicting a future one), (b) normal seasonal behavior, (c) replacing attrition, or (d) occasionally presaging a specific win. There is no mechanism to distinguish pre-announcement hiring from post-award ramp-up. The `hiring_blitz_detected` signal has look-ahead bias baked in unless job posting timestamps are verified to precede contract awards.

**Is this prediction or post-diction?** The `_check_historical_wins` function at line 242 queries Qdrant for past contract awards to the same company. This is a base rate signal (Lockheed wins defense contracts constantly) presented as predictive edge. It is not edge — it is a prior probability masquerading as a signal. A company that has won 20 contracts historically is not more likely to win this specific contract than the base rate would predict.

---

### 1.3 Convergence Alerter

**File:** `mae_core/market/intelligence/convergence_alerter.py`

**The core convergence logic and confidence boost (lines 292-297):**
```python
avg_confidence = sum(s.confidence for s in converging_signals) / len(converging_signals)
confidence_boost = min(0.2, 0.05 * (cross_domain_count - 1))
final_confidence = min(0.95, avg_confidence + confidence_boost)
```

**The independence assumption is violated.** The entire premise of multi-domain convergence boosting confidence is that signals from different domains are independent sources of evidence. If they are correlated, combining them produces false certainty. The critical flaw: many of MIDGE's domains are structurally dependent:

- The "insider" domain (Form 4 purchases) and the "congress" domain (congressional trades) both respond to the same underlying information: advance knowledge of corporate events. When both fire simultaneously, it may indicate a single leak channel, not two independent confirmations.
- The "contracts" domain (USASpending) and the "government" domain both measure government spending. A large defense appropriation will simultaneously fire signals in both categories, creating the appearance of cross-domain convergence where there is really a single causal event.
- The "sentiment" and "reddit" domains are structurally the same phenomenon (retail investor social sentiment) just read from different sources. They are nearly perfectly correlated by definition.

The `domain_categories` mapping at lines 122-137 assigns "insider" and "congress" to the same category "behavioral" — which at least prevents them from counting as different categories. But the confidence boost applies per domain, not per category. Three domains in two categories still yields `confidence_boost = min(0.2, 0.05 * 2) = 0.10`.

**The `get_actionable_summary` confidence formula (lines 425-426):**
```python
confidence = min(0.9, 0.5 + 0.1 * len(bullish_categories) + 0.1 * avg_strength)
```

This formula can produce `confidence = 0.9` when four categories are bullish with average strength 0.4. The 0.5 baseline, 0.1 per category, and 0.1 per average strength unit are arbitrary coefficients. The formula produces values between 0.5 and 0.9 no matter what the signals actually say — there is no path to a sub-0.5 "bullish" recommendation.

**The urgency signal is a velocity magnitude threshold (lines 300-306):**
```python
if avg_velocity > 0.1:
    urgency = "immediate"
elif avg_velocity > 0.05:
    urgency = "hours"
else:
    urgency = "days"
```

Velocity is a rate of change per day, but the thresholds 0.1 and 0.05 have no documented calibration. A velocity of 0.1 means the signal strength changed by 0.1 per day — but what does that mean in market terms? There is no research cited for these cutoffs.

---

### 1.4 Correlation Tracker

**File:** `mae_core/market/intelligence/correlation_tracker.py`

**Multiple comparisons problem.** With N signals, the tracker computes N*(N-1)/2 pairwise correlations. At 10 signals that is 45 pairs. At 15 signals it is 105. The anomaly threshold of 2.5 standard deviations corresponds to roughly p < 0.012 for a normal distribution. With 105 tests and no Bonferroni correction, the expected number of false positives per update cycle is approximately 105 × 0.012 = 1.26. The system will routinely flag spurious correlations as "anomalous leading indicators."

The code at line 228: `pair.is_anomalous = abs(pair.correlation_zscore) >= self.anomaly_threshold` applies the same threshold regardless of how many pairs are tracked. This is a textbook multiple comparisons failure.

**Historical std initialization (line 223):**
```python
pair.historical_std = math.sqrt(variance) if variance > 0 else 0.1
```

When variance is zero (all correlation history values are identical), std defaults to 0.1. Any non-zero correlation will then compute a z-score of `correlation / 0.1` — if the correlation is 0.3, z = 3.0, triggering an anomaly flag. This is not a real anomaly; it is a numerical artifact from insufficient history.

**Minimum observations for anomaly detection is 3 (line 220):**
```python
if len(history) >= 3:
    pair.historical_mean = ...
```

Three observations is statistically insufficient to establish a baseline for anomaly detection. With 3 data points you cannot distinguish signal from noise.

---

### 1.5 Velocity Detector

**File:** `mae_core/market/intelligence/velocity_detector.py`

**"Anomalous" is defined circularly.** The VelocityDetector flags a signal as anomalous when `velocity_zscore >= 2.5`. The velocity_std is estimated from the same rolling 20-observation window. When a new, stable signal begins accumulating observations, the first few velocity measurements dominate the distribution. A velocity that is 2.5x the rolling average of only 5-10 prior velocities is not statistically meaningful.

**The acceleration shift detector (lines 233-243) requires only 3 recent velocities.** An "inflection point" detected from 3 velocity measurements is noise, not signal. Any process with even mild random variation will generate sign changes in adjacent velocity measurements regularly.

**No stationarity check.** The velocity detector assumes the signal's statistical properties (mean, variance) are stable. Market signals are explicitly non-stationary. A signal that was quiet for 18 months and then suddenly active will show anomalous velocity — correctly — but the historical_mean and historical_std estimated from the quiet period are not valid baselines for the active period. There is no mechanism to detect distribution shifts and reset.

---

## 2. Confidence Calibration Audit

**The fundamental problem:** Confidence values in MIDGE are proxies constructed from handcrafted formulas, not empirically calibrated probabilities.

**Inventory of hardcoded confidence values with no empirical basis:**

| Location | Value | Basis |
|---|---|---|
| `signal.py:131` — `from_insider_trade()` | `confidence=0.70` | None documented |
| `signal.py:296` — `from_government_contract()` | `confidence=0.75` | None documented |
| `signal.py:337` — `from_contract_opportunity()` | `confidence=0.40` | None documented |
| `house_stock_watcher.py:72` — `CongressionalTrade` default | `confidence=0.65` | None documented |
| `cluster_detector.py:100` — `ClusterSignal` default | `confidence=0.70` | Derived from invented formula |
| `politician_tracker.py:325` — `_check_insider_pattern()` | `confidence = min(0.3 + ..., 0.7)` | None documented |
| `contract_predictor.py:81-110` — `_calculate_confidence()` | Additive 0.20+0.25+0.20+0.15 | None documented |

**The Thompson Sampler does not calibrate these values.** The Thompson Sampler learns which signal sources produce outcomes more often than others. It does not calibrate the confidence scores themselves. The calibration subsystem described in `learning_config.py` at lines 55-60 defines thresholds but there is no code in the repository that implements the calibration adjustment logic — it is defined as configuration but never executed.

**The outcomes.jsonl data reveals a calibration problem in the existing record.** The file contains 4 unique predictions replicated many times (the same prediction_ids appear across dozens of rows, indicating the evaluation loop runs the same predictions repeatedly without deduplication). After filtering to unique predictions:

- AAPL bullish, confidence 0.75: entry $185.5, outcome $278.12 — 49.9% return, marked correct
- MSFT bearish, confidence 0.60: outcome shows +1.19% initially (incorrect), then -4.49% (correct depending on run)
- LMT bearish, confidence 0.855: outcome +8.88% (incorrect — wrong direction)
- AAPL bullish, confidence 0.77: +6.99% (correct)

**Critical finding:** The AAPL "bullish" prediction with entry at $185.5 and outcome at $278.12 represents a 49.9% gain — and this same prediction appears 8+ times in outcomes.jsonl with identical values. This is not learning from 8 different AAPL predictions. It is the same prediction being re-evaluated and written to the log on every run. The Thompson Sampler is being updated multiple times with the same outcome. If sec_edgar gets 8 "success" votes from one AAPL trade, its distribution becomes artificially inflated. The forgetting mechanism in `thompson_sampler.py:379-407` decays both alpha and beta at 0.99 per call, but if updates are being duplicated at 8x the correct rate, forgetting is far too slow to counteract.

**What the distributions actually mean:** Looking at `thompson_distributions.json`:
- `sec_edgar`: alpha=1.684, beta=1.0 → mean = 1.684/2.684 = 0.627
- `form_4`: alpha=1.595, beta=1.0 → mean = 0.615
- `reddit`: alpha=1.0, beta=1.241 → mean = 0.446

These means are the seeded priors from `learning_config.py`'s reliability scores (sec_edgar=0.95 → alpha=1.9, but the stored value is 1.684 suggesting some updates have occurred). The distributions have between 0.684 and 0.241 total observations above their prior (alpha+beta-2). This represents fewer than 1 meaningful update per signal. These distributions contain almost no real learned information.

---

## 3. Data Quality and Survivorship Bias

### 3.1 Congressional Trade Data

**The 45-day lag is the defining structural problem.** The STOCK Act requires disclosure within 45 days of the trade. `house_stock_watcher.py` fetches data with a `days` parameter applied to `disclosure_date` (line 329: `if not disc_date or disc_date < cutoff`). This means MIDGE is correctly tracking when disclosures become public — but the trade itself occurred up to 45 days earlier.

**What happens to the alpha:** Academic research (CEPR, "Political power and profitable trades in the US Congress") found that before the STOCK Act, senators earned ~9.5% abnormal annual returns. After the Act, evidence of broad outperformance disappeared for rank-and-file members, though leadership positions retained some advantage. The 45-day lag means that by the time MIDGE sees a congressional trade, the market has had 0-45 days to absorb the information through other channels (earnings, contracts, news). For large-cap stocks with high analyst coverage, the information advantage is minimal to zero by day of disclosure.

**The signal assigned to congressional trades in `signal.py:204`:**
```python
event_dt = _ensure_datetime(trade.transaction_date)  # When the trade occurred
received_dt = _ensure_datetime(trade.disclosure_date)  # When MIDGE learned of it
```

This is correctly implemented — MIDGE's timestamp correctly distinguishes event time from receipt time. However, the `OutcomeTracker` uses `signal_date` (the prediction timestamp, which is `received_at`, the disclosure date) as the entry price for outcome evaluation. This means MIDGE is evaluating performance from the disclosure date forward — which is the correct and conservative thing to do — but it also means the evaluation window is measuring information that is already public, not the informational content of the original trade.

**For sector-focused members (Armed Services committee on defense stocks, Energy committee on energy stocks), the CEPR research does show persistent alpha for leadership roles.** The `politician_tracker.py` correctly focuses on committee oversight matches as a confidence booster — this is the academically supported signal. But the tracking of the specific names at lines 124-146 is a minimal hardcoded fallback list of just 4 politicians. The `congress_members.json` file loaded at line 104 is a fuller list, but the quality of committee assignment data is unverified and committee memberships change with every Congress.

### 3.2 SEC Form 4 Data — Transaction Code Contamination

**The signal file reveals a severe false positive problem in the raw data.** Looking at `data/midge/signals/2026-02-22.jsonl`:

- Lines 1-98: The overwhelming majority of signals are transaction type "D" (disposition/delivery)
- Sundar Pichai alone accounts for 14+ bearish signals in a single day on GOOGL, all from planned RSU vesting
- NVDA CFO Kress Colette accounts for 15+ bearish NVDA signals, all from planned option exercises
- The bearish signal for GOOGL at line 37 shows `shares: 676955` — this is a mass RSU delivery, not a sell decision

**The adapter in `signal.py:108-115` uses dollar value to set strength:**
```python
if is_buy:
    strength = min(1.0, trade.total_value / 1_000_000)
else:
    strength = min(1.0, trade.total_value / 500_000)
```

Pichai's sale of 676,955 shares × $333 = ~$225M produces `strength = min(1.0, 225,000,000/500,000) = 1.0`. A scheduled RSU delivery gets maximum bearish strength. This signal is noise with a confidence of 0.70 and strength of 1.0.

**The cluster detector filters for transaction code "G", "F", "M" but not "D" used for RSU delivery/derivative exercise.** Transaction code "A" (grant/award) and "D" are not equivalent. The cluster filter should also exclude code "A" (awards are not purchases) but it only filters disposition types. Looking at line 98 in the signal file: GD shows a code "A" transaction classified as "bullish" — this is a compensation grant, not an open-market purchase expressing conviction.

### 3.3 Job Posting Data

**Academic evidence on hiring signals is mixed and context-dependent.** A University of California, Irvine study (Lourie, Shevlin) found hiring data correlated with future stock performance, but the research examined aggregate job posting levels, not sudden spikes relative to industry baselines. The `job_tracker.py` looks for "spike_ratio" (30-day vs. 7-day comparison), which is a noisier signal than level-based analysis.

**The 3x spike ratio threshold in contract prediction is arbitrary.** `contract_predictor.py:91-95` gives the maximum confidence boost when `hiring_spike_ratio >= 3.0`. No academic citation or backtest validates this threshold specifically for contract prediction. Companies frequently spike hiring for reasons unrelated to contract wins: attrition spikes, new product lines, geographic expansion, or ramping up for a contract they won 6 months ago.

### 3.4 Survivorship and Lookback Bias in the outcomes.jsonl

**The outcomes data has a critical structural defect.** The same prediction_id (`5f6b90d7-41ca-425a-8553-a6d1136ac89f`) appears 8 times with identical outcome data. This is not 8 predictions — it is one prediction evaluated and written 8 times. The Thompson Sampler is being updated as though it received 8 independent confirmations. This creates artificial inflation of the `sec_edgar` source's alpha and beta values.

The `check_pending_outcomes()` method in `outcome_tracker.py` at line 165 checks `pred.get("evaluated", False)` but the prediction records in `predictions.jsonl` are never marked as evaluated — the evaluated flag is only set during the evaluation, but the record removal logic at lines 222-225 should remove evaluated predictions from the file. The fact that duplicates appear in outcomes.jsonl suggests the predictions.jsonl was not being cleared between runs during early development, and the system re-evaluated the same predictions on each run.

---

## 4. Overfitting and Data Snooping Risks

### 4.1 Thompson Sampler — Sample Size Crisis

The Thompson Sampler's distributions are seeded with priors from `learning_config.py` source reliability scores (sec_edgar=0.95, form_4=0.90, etc.). These reliability scores are author-assigned estimates, not measured values. With `prior_scale=2`, sec_edgar starts at Beta(1.9, 0.1). After the seeded initialization and decay to current state, `alpha=1.684, beta=1.0`. The `BetaDistribution.samples` property returns `int(alpha + beta - 2) = int(0.684) = 0` for most signals.

These distributions have zero meaningful real-world observations. They express the author's prior beliefs, not learned reliability. Calling `get_rankings()` and presenting these as learned reliability scores is misleading.

**The minimum samples required for meaningful Beta posteriors:** Statistical literature suggests at minimum 10-20 binary outcomes to produce a posterior that meaningfully departs from the prior. At 50+ outcomes, the distribution begins to reliably reflect reality. MIDGE currently has fewer than 5 real outcome evaluations total. The system is running a Bayesian learner with essentially no data.

### 4.2 Multiple Signal Types — Degrees of Freedom

MIDGE tracks 12+ signal types across 14+ domains. Each signal type has its own Thompson distribution plus up to 4 regime-specific sub-distributions, giving potentially 60+ parameters. With fewer than 10 real outcome observations, the system is effectively fitting 60+ free parameters to 10 data points. This is a textbook overfitting scenario. Any apparent "learned" signal reliability reflects the seeded priors, not real performance.

### 4.3 The Discovery Log Is Not Learning

The `discovery_log.jsonl` records convergence events. Lines 1-8 show cycles with "patterns_found" and "patterns_selected" but looking at the cycle summaries:
- "3/4 outcomes correct" appears across multiple runs — the same 4 outcomes checked each time
- `predictions_made: 2` appears consistently, but these are the same 2 predictions (AAPL, MSFT) being re-made

The discovery loop is re-running the same patterns on every cycle rather than genuinely accumulating new learning. The "3/4 outcomes correct" success rate is the 4-observation sample from the hardcoded test predictions — not a meaningful statistical result.

### 4.4 Regime Classifier — Limited Regimes, No Regime Persistence

**File:** `mae_core/market/intelligence/regime_classifier.py`

The regime classifier uses a 20-day SPY return and 20-day annualized volatility to assign one of 4 states. This is a simplification. The key weaknesses:

1. **2% threshold for bull/bear (line 29: `_TREND_THRESHOLD = 0.02`) is arbitrary.** A 20-day SPY return of +1.9% goes to "sideways" while +2.1% goes to "bull." This boundary produces regime switching on irrelevant daily variation.

2. **Volatility takes precedence (line 91):** When annual_vol > 25%, the regime is "volatile" regardless of trend direction. This means a strongly trending bull market with elevated volatility is classified as "volatile" and tracked separately. Regime-specific Thompson distributions for "volatile" will be entirely separate from "bull" and will require their own training samples to converge.

3. **No regime persistence check.** A one-day spike in volatility reclassifies the regime, meaning Thompson distributions get split across regime labels that change daily. An "insider_buy" signal that fires on day 1 under "bull" and day 2 under "volatile" due to a single bad VIX day updates two different distributions, diluting learning.

4. **The regime classifier only operates on SPY.** Individual stock regimes can diverge dramatically from the index. A stock in a bear trend while SPY is in a bull market would have its signals evaluated under "bull" regime distributions, which are the wrong baseline.

---

## 5. Feedback Loop Integrity

### 5.1 The Evaluation Window

`OutcomeTracker.min_price_move_pct = 2.0` (line 67). Success is defined as: the stock moved at least 2% in the predicted direction within `outcome_window_days` (default 14 days).

**2% is the wrong success metric.** This threshold measures whether any 2% move occurred, not whether MIDGE's prediction was useful. Historical SPY volatility implies any individual stock has roughly a 50-65% chance of moving ±2% within 14 days regardless of any signal. The success threshold is so low that a purely random signal system would achieve ~50% success rate. MIDGE's "3/4 outcomes correct" discovery log result (75%) is not impressive at a 2% threshold on a 14-day window — it barely exceeds what chance predicts.

**The direction-agnostic fallback (lines 283-289):**
```python
else:
    # No direction specified — magnitude alone determines success
    direction_ok = True
```

When no direction is specified, MIDGE counts any 2% move as a success. A signal that says "something will happen to MSFT" gets credit if MSFT moves 2% in either direction. This is not a prediction of edge — it is evidence collection for a random oracle.

### 5.2 Silent Failure Modes

**Qdrant as the signal store has no data integrity validation.** `memory.py` stores signals with `_point_id` derived from MD5 of `signal_id`. If a signal is stored twice (due to re-ingestion), the second write overwrites the first (Qdrant upsert semantics). There is no deduplication check before storage. The `date >= cutoff` string comparison at `memory.py:225` is fragile — it relies on ISO 8601 format. If a signal is stored with a different timestamp format, the client-side filter silently drops it.

**Missing ticker resolution for contract signals.** `signal.py:296` creates a contract award signal with `symbol=""`. The `OutcomeTracker` at line 169 checks `if not outcome_symbol` and logs a warning and skips the prediction. Government contract awards — one of MIDGE's core edge signals — silently fail the feedback loop entirely because they have no ticker. No outcome is ever recorded for contract_award signals. The Thompson Sampler never learns whether contract award signals predict stock returns. The feedback loop is broken for this entire signal class.

**Price fetcher cascade failure.** If `get_historical_price` returns None for an entry price, the outcome evaluation returns None and the prediction is kept in the queue forever (lines 181-184):
```python
if price_result is None:
    # Price unavailable — keep for retry
    remaining_predictions.append(pred)
    continue
```

Weekend dates, holidays, and delisted stocks all produce None. Predictions for stocks that delist stay in predictions.jsonl permanently, accumulating until the file grows without bound.

---

## 6. Regulatory and Legal Risk

### 6.1 Alternative Data and MNPI Exposure

The SEC's 2022 Risk Alert on MNPI (Investment Adviser MNPI Compliance Practices) specifically flagged alternative data usage without documented compliance controls as a deficiency category. MIDGE uses:
- SEC EDGAR data: fully public, no MNPI risk
- Congressional trade data: public after disclosure, no MNPI risk for the disclosed data
- SAM.gov contract opportunities: public procurement data, no MNPI risk
- USASpending contract awards: public data, no MNPI risk
- Job posting data via RapidAPI: **potential risk area**

**The job posting data risk:** RapidAPI job aggregation pulls from multiple sources including company career sites. If a company posts positions that reveal non-public strategic information (a product line, an acquisition target, a government program before public announcement), acting on this before the information becomes public could constitute MNPI trading. This is a gray area that has been the subject of SEC enforcement. MIDGE has no documented compliance review of job data sources.

### 6.2 Congressional Trade Pattern Matching

**The `politician_tracker.py` is designed to detect committee-oversight correlation patterns.** The research it is doing — "politician X on committee Y bought company Z before agency Y awarded contract to company Z" — is exactly the pattern that the SEC and DOJ have criminally prosecuted under securities fraud theories, even though the STOCK Act civil penalties are weak. If MIDGE's signals are used to trade on patterns that a prosecutor could characterize as "trading on information obtained through a government position," the user of MIDGE has exposure under SEC Rule 10b-5 regardless of whether the data source is technically public.

**The 45-day disclosure lag cuts both ways.** While it eliminates the ability to front-run the disclosure itself, trading on the disclosed information while knowing the specific committee oversight connection is trading on a mosaic that could be characterized as exploiting non-public context (the politician's committee access is public, the specific contract-triggering knowledge is not).

### 6.3 Automated Signal Systems Without Written Policies

Section 204A of the Advisers Act requires written policies to prevent MNPI misuse. Any person using MIDGE for actual trading without documented compliance procedures, a vendor due diligence review of each data source, and a legitimate business purpose for each alternative data type is running a compliance risk. The codebase contains no compliance documentation, no data source risk assessment, and no "information barrier" controls.

---

## 7. Adversarial Scenarios

### 7.1 Flash Crash / High-Velocity Intraday Event

MIDGE's signal ingestion window is 72 hours (the `convergence_window_hours` default in `convergence_alerter.py:98`). All signals within 72 hours contribute to a convergence alert. A flash crash intraday produces: simultaneously high velocity readings, elevated bearish sentiment, and potentially triggering VelocityDetector anomaly flags — which then gets flagged as an "immediate urgency" convergence signal.

The system would generate a bearish "IMMEDIATE" convergence alert at the bottom of a flash crash, potentially triggering a sell at the worst possible moment. The deduplication window of 4 hours (`_min_alert_interval_hours`) would suppress subsequent recovery signals for 4 hours.

### 7.2 Correlated Signal Collapse — The "Everything Goes Wrong" Scenario

MIDGE's convergence logic counts domain-level signals. If a single piece of news triggers simultaneous signals across insider (executives pre-position), congress (congressional trades disclosed), contracts (agency budget announced), and sentiment (retail reacts) domains, MIDGE generates a high-confidence convergence alert. But if the underlying event is already priced in by the time all four signals accumulate, MIDGE is chasing a move that has already happened.

The 72-hour window means signals from 3 days ago count equally with signals from 1 hour ago. A convergence pattern built from a 3-day-old insider filing and a 2-day-old congressional disclosure and a 1-hour-old news event is presented as a unified fresh signal.

### 7.3 Regime Change Not in the Training Set

The regime classifier has never seen a hyperinflation regime, a sustained currency crisis, or a zero-bound interest rate environment (though the latter existed historically). The Thompson distributions learned under the current market regime will have incorrect calibration during a regime change. There is no out-of-distribution detection, no regime transition alert, and no mechanism to flag "this is unlike anything the system has seen before."

### 7.4 Adversarial Use Against MIDGE

Congressional trade data is published by `unusualwhales.com`, `capitoltrades.com`, and others. Any sufficiently motivated actor can construct fake signals by making trades that will be disclosed, triggering MIDGE's patterns, and front-running MIDGE's expected trades. The politician tracker uses a `KNOWN_POLITICIANS` dictionary — a sophisticated adversary who knows this list can manufacture correlations by making small trades through known politicians' disclosed portfolios. The verification pathway (Mae's Law 1) does not protect against this because all three witnesses can be poisoned by the same public data stream.

---

## 8. What MIDGE Gets Right

### 8.1 The Multi-Domain Concept Has Genuine Theoretical Support

The core insight — that signals from genuinely independent domains provide stronger evidence than single-domain signals — is supported by multi-factor investment literature (Fama-French factors work precisely because they are partially orthogonal) and information theory (independent evidence compounds probability more than correlated evidence). The implementation failure is not in the concept but in the failure to verify signal independence before combining.

### 8.2 The Form 4 Signal Has Real Academic Support

Insider purchases — specifically large, open-market code "P" purchases by officers and directors — do predict positive returns. The Jeng et al. (2003) finding of ~6% annual abnormal returns is the floor. For small-cap and micro-cap stocks (where MIDGE's watchlist does not focus), the signal is stronger. A 2026 arXiv paper on microcap Form 4 purchases found AUC of 0.70 on out-of-sample data, suggesting meaningful predictive content. The problem is that MIDGE's implementation contaminates this signal with planned transactions.

### 8.3 The Feedback Architecture Is Correctly Designed in Principle

`OutcomeTracker` → `ThompsonSampler.update()` is the right architecture for a Bayesian learning system. The causal chain is correct: generate prediction, observe outcome, update belief. The Thompson Sampling exploration/exploitation balance is a principled approach to signal selection. These are good architectural choices. The implementation failures (duplicate outcomes, no ticker for contract signals, 2% threshold too low) are bugs, not design errors.

### 8.4 The Decay Rate System Is Sound

The half-life system in `learning_config.py` (news: 1.4 days, insider: 14 days, contract: 35 days) correctly models the information persistence of different signal types. This is consistent with academic literature on signal decay (insider signals do persist longer than news signals). The implementation in signal decay calculations is technically correct.

### 8.5 The Politician-Contract Correlation Is a Genuine Edge — For a Narrow Condition

When a politician on a specific oversight committee makes a large open-market purchase in a company that company's specific agency then awards a contract to within 30 days — that correlation has genuine informational content. The CEPR research ("Political power and profitable trades") shows leadership roles retain statistically significant abnormal returns even post-STOCK Act. This is MIDGE's sharpest, most defensible signal. It is also the rarest.

---

## 9. Priority Recommendations for the Build Team (Witness Beta)

These are not recommendations for immediate implementation — they are findings for Beta to prioritize and propose:

1. **Fix the outcome duplication bug.** The same prediction IDs appearing 8x in outcomes.jsonl means Thompson Sampler distributions are being trained on fabricated data. This is the highest priority because it corrupts the entire learning layer.

2. **Filter transaction codes properly.** Exclude code "D" (delivery for RSU/option vesting), "A" (award/grant), "V" (10b5-1 plans) and any transaction where the filer shows a pattern of same-date multi-tranche executions (automated plan signature). This will dramatically reduce the signal-to-noise ratio in the insider layer.

3. **Establish a proper null model.** Before claiming any signal has edge, measure its hit rate against a 50% random baseline and against simple momentum. The 2% success threshold at 14 days is not sufficient — use 5% or alpha-vs-SPY.

4. **Fix the missing ticker problem for contract signals.** The feedback loop is silently broken for contract_award signals. Either resolve the ticker at ingestion time or exclude unresolved signals from outcome tracking.

5. **Add multiple comparisons correction to CorrelationTracker.** Apply Bonferroni or Benjamini-Hochberg correction: `adjusted_threshold = 0.005 / n_pairs` rather than a fixed 2.5 sigma.

6. **Replace invented confidence constants with calibrated values.** The 0.70 base for insider signals, the 0.65 for congressional signals — these should come from empirical calibration against outcomes, not from author assertions.

---

## Sources

Academic research cited:

- [Insider Purchase Signals in Microcap Equities (arXiv 2602.06198)](https://arxiv.org/abs/2602.06198)
- [Insider filings as trading signals — Does it pay to be fast? (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1544612324015435)
- [The Death of Insider Trading Alpha (SSRN 5966834)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5966834)
- [Do senators and house members beat the stock market? (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0047272722000044)
- [Political power and profitable trades in the US Congress (CEPR)](https://cepr.org/voxeu/columns/political-power-and-profitable-trades-us-congress)
- [Thompson Sampling for Non-Stationary Bandit Problems (MDPI)](https://www.mdpi.com/1099-4300/27/1/51)
- [A Tutorial on Thompson Sampling (Stanford)](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)
- [Job postings and aggregate stock returns (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1386418123000022)
- [Can Alternative Data be Considered Insider Trading? (Acuity)](https://acuitytrading.com/news-story/can-alternative-data-be-considered-insider-trading)
- [SEC Risk Alert: MNPI Compliance Practices (SEC.gov)](https://www.sec.gov/files/code-ethics-risk-alert.pdf)
- [Congressional Stock Trading and the STOCK Act (CLC)](https://campaignlegal.org/update/congressional-stock-trading-and-stock-act)
- [What explains trading behaviors of members of congress? (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1059056024005835)
- [Congress Trading Report 2024 (Unusual Whales)](https://unusualwhales.com/congress-trading-report-2024)
