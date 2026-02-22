# Witness Beta Findings: What Breaks

**Lens: Adversarial — Tuning, Failure Modes, and What Breaks**

---

## Executive Summary

The market modules are a thoughtfully structured skeleton with serious problems in calibration, validation, and integration readiness. The code will not produce garbage under ideal conditions — but ideal conditions will never exist. The primary failure categories are: learned distributions that are mathematically overconfident before any real data exists, a Thompson Sampler that has no forgetting mechanism and will lock onto early noise permanently, a ConvergenceAlerter that can fire on two signals from the same company appearing in two domains, and API clients that silently return empty results on failure in ways that will produce confident-looking zero-signal states rather than honest error states.

---

## SECTION 1: Parameter Audit

### 1.1 ThompsonSampler — `mae_core/market/intelligence/thompson_sampler.py`

**DEFAULT_PRIOR_SCALE = 10** (line 32)

This is the first and largest problem. When `_seed_from_reliability()` runs, it converts a reliability score `r` into `Beta(r * 10, (1-r) * 10)`. For `sec_edgar` with reliability 0.95, that produces `Beta(9.5, 0.5)` — a distribution with mean 0.95 and variance approximately 0.004. This distribution is extremely tight. After only 10 real observations, this prior has the same weight as 10 actual outcomes. The sampler treats the manually assigned 0.95 reliability score as if it were derived from 10 historical data points. It is not — it is an assumption.

The distribution file confirms this is exactly what was seeded (line 3-7 of `thompson_distributions.json`): `sec_edgar` has `alpha=15.5, beta=1.5`, a mean of 0.912, variance ~0.005. This was seeded from prior_scale=10 (the default) applied to reliability=0.95 which produces the 9.5/0.5 values shown, then some updates happened to get to 15.5/1.5 — suggesting 6 "successes" were logged, but against what ground truth is unknown.

**Assessment:** DANGEROUS. `prior_scale=10` makes the system believe it has 10 real observations before any real data is collected. For a system with no outcome history, `prior_scale=2` or even `prior_scale=1` would be more honest. The appropriate prior scale should be determined by how many real observations would be needed to meaningfully shift the belief — for financial signal reliability, you would want at least 50-100 real outcome samples before the prior becomes informative.

**Recommendation:** Set `DEFAULT_PRIOR_SCALE = 2`. Allow it to be overridden per-deployment once real data accumulates.

---

**No Decay / Forgetting Mechanism** (entire `update()` method, lines 197-254)

The Thompson Sampler has no time-based decay. Every historical outcome has equal weight forever. A signal that was reliable in 2023 but is now broken (API changed, market regime changed, pattern mined away) will never be forgotten. Alpha and beta only ever increase. A signal with 500 observations that has been dead for 6 months still influences sampling at full weight.

**Assessment:** CRITICAL INTEGRATION HAZARD. When wired into live data, stale reliability scores will persist indefinitely. The system will exploit signals that have stopped working.

**Recommendation:** Implement Bayesian forgetting via multiplicative decay: on each update, apply `alpha *= decay_factor; beta *= decay_factor` before the new observation (where `decay_factor` is regime-specific from `learning_config.py`'s `decay_rates`). This brings total weight toward the prior over time while preserving the mean direction. Minimum: apply decay daily.

---

**`BetaDistribution.samples` property** (line 52-55)

```python
return int(self.alpha + self.beta - 2)  # Subtract initial prior
```

This assumes the prior was Beta(1,1) for everything. But seeded signals start at Beta(9.5, 0.5). The reported "samples" for `sec_edgar` (alpha=15.5, beta=1.5) would be reported as `int(15.5 + 1.5 - 2) = 15`, implying 15 real observations. The actual number of post-seeding updates might be 6 (the 15.5 - 9.5 = 6 alpha increment). This misleads any diagnostic code that uses `.samples` to gate decisions.

**Assessment:** MISLEADING. If any downstream code gates on `dist.samples >= N`, it will activate prematurely for seeded signals.

**Recommendation:** Store `observation_count` separately from alpha/beta. Do not derive sample count from distribution parameters when the prior is non-uniform.

---

**`get_uncertain_signals()` — `min_variance: float = 0.01`** (line 311)

Beta(9.5, 0.5) has variance approximately `(9.5 * 0.5) / ((10^2) * 11)` = 0.00432. This is below the 0.01 threshold. The seeded high-reliability signals will NEVER appear as "uncertain" even though they have zero real validation. The system will not explore these signals.

**Assessment:** The min_variance=0.01 threshold is calibrated against uninformative priors Beta(1,1) (variance=0.25) but is wrong for pre-seeded signals. `sec_edgar` starts with variance 0.004 — below threshold — despite having no real validation at all.

**Recommendation:** Lower default to `min_variance=0.001`, or derive the threshold adaptively based on the distribution's alpha+beta total weight relative to `prior_scale`.

---

### 1.2 learning_config.py — `mae_core/market/intelligence/learning_config.py`

**`decay_rates` are defined but never used** (lines 23-32)

The `learning_config.py` defines per-signal-type decay rates. The `ThompsonSampler` never reads them. There is no mechanism connecting the config's decay rates to the sampler's update logic. The decay rates exist as documentation only.

**Assessment:** DEAD CODE creating false confidence. The CLAUDE.md says "Self-modifiable learning parameters" — but modifying these does nothing because nothing reads them.

---

**`source_reliability` defaults** (lines 36-49)

The defaults themselves are reasonable opinions — SEC EDGAR at 0.95 is defensible, Reddit at 0.30 is defensible. The problem is the `unknown` signal at 0.50 means an unknown signal type is treated as coin-flip reliable, which is more optimistic than warranted for truly unknown provenance. More critically, these reliabilities apply globally to the source, not to the signal type + direction + market regime combination. An SEC Form 4 "buy" signal in a bear market has a very different reliability than the same signal in a bull market. The flat 0.95 for `sec_edgar` ignores all of this context.

**Assessment:** ACCEPTABLE as starting points but will produce poorly calibrated outputs for regime-specific use cases.

---

**`overfitting_protection.min_samples = 50`** (line 62)

This is the minimum predictions before adjusting reliability. 50 predictions at even daily operation could take months to accumulate for low-frequency signals. Until then, no adjustment happens.

**Assessment:** REASONABLE for protection against overfitting, but means the system is essentially static for the first months of operation. Fine for launch, concerning for anyone expecting rapid adaptation.

---

**`confidence_calibration` thresholds** (lines 53-55)

The confidence calibration defines `underconfident_threshold: 0.6` and `overconfident_threshold: 0.4` but there is no code in any of the five intelligence files that reads or applies these values. Like the decay_rates, this is aspirational infrastructure with no implementation.

**Assessment:** DEAD CONFIG. Flag for implementation.

---

**`update_config()` mutates a module-level dict at runtime** (lines 94-143)

The `LEARNING_CONFIG` dict is module-level. If two threads or two agents call `update_config()` simultaneously, there is a race condition on the dict mutation and the version increment. The function also logs to a hardcoded relative path `Path(__file__).parent / "config_history.jsonl"` — this is a file inside the market module directory, not in `data/market/`. This is inconsistent with where `thompson_sampler.py` writes its history (`DATA_DIR` = project root `/data/market/`).

**Assessment:** INTEGRATION HAZARD. When wired into Mae's multi-agent environment, concurrent config updates will produce data corruption.

**Recommendation:** Use a proper mutex or make config updates atomic via a dedicated config agent. Move the history log path to `DATA_DIR` for consistency.

---

### 1.3 ConvergenceAlerter — `mae_core/market/intelligence/convergence_alerter.py`

**`min_domains = 2` default** (line 89)

The core value proposition of the convergence alerter is described as "3+ domains from different categories = high confidence." The default minimum is 2. Two signals from the same company, one recorded under "insider" domain and one under "government" domain (a contract award to the same company), would trigger a convergence alert. These are not independent signals — they are correlated by definition (the same company is in both). The convergence alert would fire on what is essentially a single compound event, not true cross-domain confirmation.

**Assessment:** DANGEROUS. The min_domains=2 default will produce alerts that look like convergence but represent single-entity correlation.

**Recommendation:** Set default `min_domains=3` to require at least 3 independent domains, matching the documented design intent and Mae's Law of Triadic Generator.

---

**`confidence_boost = min(0.2, 0.05 * (cross_domain_count - 1))`** (line 255)

Maximum confidence boost from domain convergence is 0.20 (at 5+ categories). Base confidence of 0.5 + 0.20 = 0.70 maximum from cross-domain agreement alone, before signal strength factors. This ceiling seems reasonable. However, the boost is linear and does not account for the quality of category diversity. Five domains all categorized as "market" (crypto, technical, volume, sentiment, reddit) would provide no cross-category boost — but the current implementation would give a boost since `cross_domain_count = len(categories_seen)`. Wait — looking more carefully, `categories_seen` uses `domain_categories` map. Reddit maps to "social", sentiment maps to "social" too. So two social signals would contribute only 1 to `cross_domain_count`. This part is actually correctly implemented.

**Assessment:** The category-based boost is correctly implemented. The `min_domains` floor is the real problem, not the boost formula.

---

**`_prune_old_signals()` is called on every `record_signal()` call** (line 174)

This iterates over ALL signals in ALL domains on every insert. With 13 defined domains and potentially hundreds of signals per domain during a high-frequency feed, this becomes O(n*domains) per insertion. If signals arrive at high frequency (e.g., 100+ per second from an EventBus), this will be a performance bottleneck.

**Assessment:** SCALABILITY CONCERN. Not critical at current scale but will cause latency when wired to EventBus.

**Recommendation:** Prune on a timer (every 60 seconds) rather than on every insert.

---

**`check_convergence()` stores alerts indefinitely in `self.alerts`** (line 216)

`self.alerts.extend(alerts)` with no bound. Over days of operation, this list grows without limit. Memory leak for long-running process.

**Assessment:** MEMORY LEAK. Low severity at low alert frequency, but will accumulate.

**Recommendation:** Cap at `self.alerts = self.alerts[-1000:]` or use a deque with maxlen.

---

**No deduplication of alerts** (lines 200-218)

`check_convergence()` generates a new alert every time it is called if conditions are met. If called every minute and conditions persist for 2 hours, 120 identical alerts are generated. There is no "alert already firing" state.

**Assessment:** ALERT STORM when wired to periodic EventBus polling. Will flood downstream consumers.

**Recommendation:** Track `last_alert_direction` and `last_alert_time`. Suppress re-alerting within a minimum interval (e.g., 4 hours) unless strength changes materially.

---

**`get_actionable_summary()` confidence formula** (lines 382-383)

```python
confidence = min(0.9, 0.5 + 0.1 * len(bullish_categories) + 0.05 * bullish_strength)
```

`bullish_strength` is a sum of domain strengths, not normalized. With 5 bullish domains each with strength 0.8, `bullish_strength = 4.0`. The formula gives `0.5 + 0.1*5 + 0.05*4.0 = 1.5`, capped at 0.9. The `0.05 * bullish_strength` term is essentially always hitting the cap at any reasonable signal count. The cap at 0.9 prevents it from being truly dangerous, but the formula is poorly calibrated — signal strength is doing almost no work in distinguishing outcomes.

**Assessment:** TUNING ISSUE. The strength term contributes meaninglessly because it almost always pushes past the cap.

**Recommendation:** Normalize `bullish_strength` by domain count: `avg_strength = bullish_strength / len(bullish_domains)`. Then `confidence = min(0.9, 0.5 + 0.1 * len(bullish_categories) + 0.1 * avg_strength)`.

---

### 1.4 VelocityDetector — `mae_core/market/intelligence/velocity_detector.py`

**Velocity computed in units of /second** (lines 132-136)

```python
dv = value - state.last_value
velocity = dv / dt   # dt is in seconds
```

If a signal value represents "number of insider buys" and observations come daily, `dt` is 86400 seconds. A change from 2 to 8 buys yields velocity = `6/86400 = 0.0000694 per second`. Then z-score thresholds and anomaly detection operate on these sub-thousandth values. The velocity_std for historical velocities computed from the same scale will also be tiny. This doesn't cause mathematical errors but makes all velocity values hard to reason about and interpret. The `get_leading_signals()` check `if avg_velocity > 0.1:` (in ConvergenceAlerter) will NEVER trigger for daily-frequency signals.

**Assessment:** INTEGRATION HAZARD. The convergence alerter's urgency classification uses `avg_velocity > 0.1` as the threshold for "immediate" urgency. At daily frequency, velocities are ~0.0001 scale. Urgency will always read "days" regardless of actual acceleration.

**Recommendation:** Normalize velocity to units-per-day rather than per-second. Replace `dt = (timestamp - state.last_timestamp).total_seconds()` with `dt = (timestamp - state.last_timestamp).total_seconds() / 86400`.

---

**`window_size = 20` default** (line 70)

For signals observed once per day, a window of 20 means 20 days of data are needed before anomaly detection becomes meaningful. For SEC EDGAR (which generates Form 4 signals sporadically), 20 observations could take months to accumulate for a single ticker. For signals observed multiple times per day (price, volume), 20 is too small to establish a meaningful baseline.

**Assessment:** TUNING ISSUE. The same window size is inappropriate for all signal types. Should be configurable per signal category.

**Recommendation:** Allow window_size to be passed per-signal via `record()`, or maintain a config map of window sizes by signal type.

---

**`_update_velocity_stats()` uses population variance** (line 173)

```python
variance = sum((v - state.velocity_mean) ** 2 for v in velocities) / len(velocities)
```

This is population variance (divides by N). For small samples (2-5 observations), this will systematically underestimate variance and produce z-scores that are too high, flagging too many anomalies early in a signal's life.

**Assessment:** STATISTICAL ERROR for small windows. Should use Bessel's correction (divide by N-1) for sample variance. Low severity because the `min_observations` guard at line 207 prevents anomaly flagging until 5+ observations, which is still small.

**Recommendation:** Replace `/ len(velocities)` with `/ max(1, len(velocities) - 1)`.

---

**`anomaly_threshold = 2.5`** (line 72)

A 2.5 standard deviation threshold assumes approximately Gaussian velocity distribution. Financial signal velocities are heavy-tailed and non-Gaussian. A 2.5 sigma threshold in a truly Gaussian distribution would fire roughly 1.2% of the time by chance. In a heavy-tailed distribution, it fires far more often.

**Assessment:** ACCEPTABLE as a starting parameter, but expect more false positives than the Gaussian assumption would predict. For production use, consider 3.0 or derive empirically.

---

**`detect_acceleration_shifts()` — inflection detection is noise-prone** (lines 229-239)

The method checks whether the last 3 velocity values show a sign change in the second difference. With only 3 points, any noisy signal will produce spurious inflection detections. If signal values have any randomness, this will fire constantly.

**Assessment:** TUNING ISSUE. Three points is far too few for reliable inflection detection. Recommend requiring at least 5 points and using a smoothed velocity (e.g., exponential moving average) before computing second differences.

---

### 1.5 CorrelationTracker — `mae_core/market/intelligence/correlation_tracker.py`

**Timestamp alignment uses O(n^2) inner loop** (lines 147-161)

```python
for ts_a, val_a in history_a:
    for ts_b, val_b in history_b:  # O(n^2) inner loop
        diff = abs(ts_a - ts_b)
        ...
```

For each signal pair with window_size=30 observations each, this is 30*30=900 operations. With N tracked signals, `update_correlations()` computes N*(N-1)/2 pairs, each with O(window^2) alignment. With 20 signals and window=30: `20*19/2 = 190 pairs * 900 = 171,000 operations per update call`. Called from `detect_correlation_anomalies()` and `get_most_correlated_pairs()` which are themselves called on demand.

**Assessment:** SCALABILITY CONCERN. At 10 signals this is fine. At 50+ signals this becomes a bottleneck, especially if `update_correlations()` is called frequently. When wired to EventBus with many signals, this will cause latency spikes.

**Recommendation:** Sort history by timestamp and use binary search for alignment (O(n log n) total). Or, more simply, resample all signals to a common time grid and use vectorized Pearson computation.

---

**`compute_correlation()` returns 0.0 if constant signal** (line 175)

```python
if var_a == 0 or var_b == 0:
    return 0.0
```

A constant signal (e.g., a sensor reporting the same value repeatedly) has zero variance. Returning correlation=0 is mathematically correct (undefined but conventionally 0) but then this gets stored, tracked, and generates z-scores. A constant signal with correlation history all 0.0 will have `historical_std=0` and produce undefined z-scores (division by zero avoided only by the `if pair.historical_std > 0` guard at line 224). However, if a signal is constant for a while and then becomes variable, its historical std will be near-zero and any correlation will appear as a massive anomaly. This is a false alarm factory for episodic signals.

**Assessment:** EDGE CASE that will produce false correlation anomalies for signals that become active after a quiescent period.

**Recommendation:** When a signal has been constant for more than half its window, exclude it from correlation tracking until it becomes variable.

---

**`correlation_history` maxlen = window_size = 30** (line 205)

The correlation history and the signal history share the same window size. This means that the "historical mean" of correlation is based on only 30 correlation readings. For daily-frequency signals, that's 30 days of historical correlation context. If the system restarts, all historical correlation context is lost (since `_load_state()` doesn't restore the deque contents, only the pair stats).

**Assessment:** CONTEXT LOSS ON RESTART. The loaded state contains the last-known stats (mean, std, zscore) but the underlying deque is empty. The first few correlations after restart will be compared against a historical mean derived from pre-restart data but with fresh history — no way to add new readings to refine the historical mean without first repopulating the deque.

**Recommendation:** Persist the deque contents to disk. The current persistence only saves the summary stats, not the raw history needed to update them correctly.

---

**`min_observations = 10` for correlation** (line 60)

Ten observations to compute a Pearson correlation is statistically laughable. With 10 points, the 95% confidence interval for a correlation of 0.6 spans roughly [0.0, 0.9]. The correlation estimate is essentially meaningless for investment decisions.

**Assessment:** STATISTICAL INADEQUACY. For reliable correlation estimates in trading contexts, a minimum of 30-50 observations is required. The 10-point minimum will produce wildly variable correlation estimates that will generate false anomalies.

**Recommendation:** Increase `min_observations = 30`. Accept that the system will be slow to activate, but the activations will be meaningful.

---

## SECTION 2: Failure Modes — API Failures and Bad Data

### 2.1 SEC EDGAR Client — `mae_core/market/apis/sec_edgar/client.py`

**Silent failure returns empty list** (lines 54-61)

```python
except Exception as e:
    print(f"SEC EDGAR request failed: {e}")
    return None
```

Every failure returns None or empty list. The calling code at `get_recent_form4s()` (lines 67-105 of `__init__.py`) proceeds silently with zero trades. The convergence alerter gets no insider signal. No error is propagated. No circuit breaker trips. From the system's perspective, "no insider buying in the last 30 days" and "API was down" look identical.

**Assessment:** CRITICAL FAILURE MODE. When integrated into bootstrap or EventBus, API failures will silently produce absence-of-signal, which the convergence alerter interprets as no evidence. This is the difference between "we have no information" and "we checked and found nothing" — the system cannot distinguish them.

**Recommendation:** All API clients should distinguish between `[]` (empty result, API worked) and `None` (API failed). Callers should check for None and raise or propagate errors through the event bus. Add a health-check method to each API client.

---

**`get_company_cik()` downloads the full ticker list on every call** (lines 63-86)

The function calls `self._get(SEC_TICKERS_URL)` which downloads the complete company tickers JSON file from SEC every time a CIK lookup is needed. This file is large (>1MB) and downloaded without caching. For `get_recent_form4s("AAPL")` followed by `get_recent_form8ks("AAPL")`, the file is downloaded twice (two separate `SECEdgarClient` instances are created). For scanning 100 symbols, this file is downloaded 200 times.

**Assessment:** PERFORMANCE BUG. The ticker JSON should be cached per-session (or to disk with a 24h TTL).

---

**`parse_form4()` has undocumented fallback logic** (lines 151-181)

The method tries multiple URL patterns for the XML document. If the primary URL returns HTML rather than XML, it switches parsing modes silently. The XML parser also silently skips transactions where price=0 or shares=0 (line 401). Options grants often have price=0 and would be silently filtered out. This is intentional design (skip options) but is not documented and has no logging.

**Assessment:** ACCEPTABLE but note that the `transaction_type` field from XML parsing is `A/D` (Acquired/Disposed) while the models claim it could be `P/buy/purchase`. The `InsiderTrade.is_purchase` attribute referenced in `contract_predictor.py` at line 232 (`trade.is_purchase`) does NOT EXIST on the `InsiderTrade` dataclass. This will raise `AttributeError` at runtime.

```python
# contract_predictor.py line 232:
if trade.is_purchase and trade.transaction_code == "P":
```

`InsiderTrade` has no `is_purchase` property. This is a guaranteed crash when `_check_insider_buying()` runs against any real data.

**Assessment:** CRASH BUG. Priority: fix before any live testing.

**Fix:** Add to `InsiderTrade` dataclass: `@property def is_purchase(self): return self.transaction_type in ("A", "P", "buy", "purchase")` — or just check `transaction_type` directly in the caller.

---

**`_parse_form4_html()` regex is fragile** (lines 270-373)

The HTML parser uses regex against SEC-rendered HTML. SEC EDGAR is known to change their HTML rendering. The regex patterns like:
```python
re.search(r'Reporting Person.*?<a[^>]*>([^<]+)</a>', ...)
```
are tied to the current XSLT rendering format. When SEC updates their rendering (which they have done historically), all HTML parsing will break silently, returning empty trade lists.

**Assessment:** FRAGILITY. Not a day-one concern but will break without warning. The XML parsing path is more robust. Recommend flagging this as a monitoring point.

---

**Contact email in user agent** (line 22)

`SEC_USER_AGENT = "MIDGE Trading Research contact@example.com"`

The `@example.com` email is a placeholder. SEC's robots.txt and their EDGAR fair access policy require a real contact email. Using a fake email violates their terms and could result in IP blocks.

**Assessment:** COMPLIANCE ISSUE before production deployment.

---

### 2.2 ClusterDetector — `mae_core/market/edge/cluster_detector.py`

**Qdrant assumed live, no connection check** (lines 172-211)

`_query_recent_trades()` connects directly to `http://localhost:6333` with a 10-second timeout. If Qdrant is not running, every cluster query fails silently and returns `[]`. The `scan_all_symbols()` function would then report "no clusters" for any symbol list when Qdrant is simply down.

**Assessment:** CRITICAL DEPENDENCY. This is the first of several modules hardcoding Qdrant URL at `http://localhost:6333`. When wired into bootstrap, these need to use the existing ConnectionRegistry or ApiGateway — not hardcoded localhost.

---

**`_aggregate_by_insider()` groups by name string** (line 246-254)

Insider names are matched as exact strings: `"John Smith"` and `"SMITH, JOHN"` and `"J Smith"` are treated as different people. SEC EDGAR Form 4 data is inconsistent in name formatting. Two entries from the same insider could appear as different people, undercount cluster size, and fail to trigger.

**Assessment:** FALSE NEGATIVE SOURCE. Name normalization (uppercase, surname-first standardization) is missing.

---

**`store_cluster_signal()` uses `abs(hash(cluster_id)) % (10**18)` as Qdrant ID** (line 620)

Python's `hash()` is non-deterministic across process restarts (since Python 3.3, hash randomization is enabled by default). The same `cluster_id` string will produce different Qdrant IDs on each restart. This means the same cluster could be stored multiple times with different IDs, creating duplicates. Qdrant point IDs must be stable for upserts to work correctly.

**Assessment:** DATA CORRUPTION BUG. Use `uuid.UUID(cluster.cluster_id).int` instead, since `cluster_id` is already a UUID string.

---

**`RelationshipTracker._record_relationship()` running average** (lines 530-538)

```python
rel.avg_time_delta_hours = (
    (rel.avg_time_delta_hours * (rel.trades_together - 1) + delta_hours)
    / rel.trades_together
)
```

This computes a running average correctly, but at line 539:
```python
rel.correlation_score = min(0.95, 0.5 + (rel.trades_together * 0.1))
```

After 5 co-trades, correlation_score = 1.0 (capped at 0.95). Five trades in a stock between two people over a year could be coincidence, not coordination. The formula is too aggressive — it reaches maximum confidence in 5 observations with no statistical validation.

**Assessment:** OVERCONFIDENT RELATIONSHIP DETECTION. A stock that happens to be volatile will generate many trades from all insiders, producing spurious "coordinated trading" relationships between people who simply react to the same news.

**Recommendation:** Add minimum time span requirement (relationships spread over 60+ days only), and require co-trades in multiple different stocks before raising score above 0.6.

---

### 2.3 PoliticianTracker — `mae_core/market/edge/politician_tracker.py`

**KNOWN_POLITICIANS dictionary has 4 entries** (lines 97-126)

The entire politician-trade edge detector depends on matching against 4 hardcoded politicians. Nancy Pelosi is no longer in a relevant position for most trades (former Speaker). The 116th Congress member list is not included. The system will silently flag zero politician trades for every name not in this list — which is essentially everyone.

**Assessment:** FUNCTIONALLY BROKEN as written. The purpose is to detect congressional trading patterns but only 4 people are in scope.

---

**`_identify_politician()` last-name matching** (lines 202-214)

```python
last_name = profile.name.split()[-1].upper()
if last_name in filer_upper:
    return profile
```

"SULLIVAN" as a last name would match any filer whose name contains "Sullivan" — including non-politicians. "SMITH" would match too broadly. This false-positive matching means regular insiders could be misclassified as politicians, triggering contract correlation searches for ordinary corporate executives.

**Assessment:** FALSE POSITIVE SOURCE. Needs full-name matching, not last-name substring matching.

---

**`_symbol_to_company()` has 11 hardcoded mappings** (lines 365-381)

The contract correlation only works for 11 specific stocks (LMT, RTX, BA, NOC, GD, MSFT, AMZN, GOOGL, AAPL, META, NVDA). Any other symbol returns the ticker itself as the company name, which will fail to match contracts in USASpending.

**Assessment:** SCOPE LIMITATION. Not a bug per se, but the module is silently useless for any stock outside these 11.

---

**Timing window: `-30 <= days_diff <= 90`** (line 251)

The trade must occur between 30 days before and 90 days after a contract award. This is extremely wide. A politician who buys a defense stock and then 3 months later the company wins a contract will be flagged. Given that major defense contractors win contracts constantly, almost any purchase would find a matching contract within a 90-day forward window. The base confidence of 0.3 plus even a small timing bonus would frequently clear the 0.5 threshold needed to return a signal.

**Assessment:** EXCESSIVE FALSE POSITIVE WINDOW. The predictive value comes from trades BEFORE the contract, not after. Restrict to trades occurring within 60 days before the contract award.

---

### 2.4 FilingTimeAnalyzer — `mae_core/market/edge/filing_time_analyzer.py`

**No timezone handling** (lines 125-135)

The analyzer classifies filings as pre-market, market hours, or after-hours based on `MARKET_OPEN = time(9, 30)` and `MARKET_CLOSE = time(16, 0)`. The `filing_datetime` parameter accepts a naive datetime. If the datetime is in UTC (as SEC EDGAR returns) but the code compares against Eastern Time boundaries, a 20:00 UTC filing (which is 15:00 ET during summer) would be misclassified as after-hours.

**Assessment:** SYSTEMATIC MISCLASSIFICATION for all non-Eastern-Time datetimes. The filing time analysis will produce wrong pattern detections until timezone awareness is added.

**Recommendation:** Convert all filing datetimes to Eastern Time (US/Eastern) before comparison. Use `zoneinfo` or `pytz`.

---

**Hardcoded `Qdrant_URL = "http://localhost:6333"` in `__init__`** (line 102)

Same issue as other modules. Not using ApiGateway or config.

---

**When no filing time is available, defaults to 4 PM ET** (line 235)

```python
filing_dt = datetime.strptime(f"{filing_date} 16:00", "%Y-%m-%d %H:%M")
```

If filing time is unknown, the code assumes 4 PM. This is right at the edge of market hours / after-hours boundary. The resulting classification might be "after_hours" (if the classification uses `>=` check) or "market_hours" (if `>`). This default assumption will apply the "after_hours" mild-suspicious modifier (-0.08) to any filing for which time data is missing, systematically biasing the analyzer toward suspicion.

**Assessment:** SYSTEMATIC BIAS for unknown-time filings.

---

### 2.5 ContractPredictor — `mae_core/market/edge/contract_predictor.py`

**`is_active_bidder=True` assumed** (line 189)

```python
is_active_bidder=True  # Assume they're bidding if we're checking
```

When analyzing a contract, every company passed in `potential_bidders` is assumed to be an active bidder, contributing +0.20 to confidence. With a minimum possible confidence of 0.20 (active bidder only, no other signals), any company checked at all immediately gets a 20% confidence of winning. With just a hiring spike detected as well, confidence reaches 0.40+. This means the system produces non-trivial confidence scores from thin evidence.

**Assessment:** BASELINE OVERCONFIDENCE. The active-bidder assumption adds 0.20 without verification. SAM.gov bid registration data is not checked.

---

**`_check_historical_wins()` uses exact name matching in Qdrant** (lines 248-268)

```python
{"key": "recipient_name", "match": {"value": company_name}}
```

Exact match only. "Lockheed Martin Corporation" won't match "Lockheed" or "LOCKHEED MARTIN". Qdrant's exact match filter is case-sensitive and will return 0 results for most lookups unless the stored names match exactly.

**Assessment:** `historical_winner` will almost always be False due to name mismatch, making this signal permanently inactive.

---

### 2.6 Job Tracker — `mae_core/market/apis/job_tracker.py`

**Spike detection: 24h jobs vs 30d average — sample size issue** (lines 302-307)

```python
daily_avg = signal.jobs_30d / 30
signal.spike_ratio = signal.jobs_24h / daily_avg
signal.is_spike = signal.spike_ratio >= 2.0
```

The `signal.jobs_7d` and `signal.jobs_30d` counts are derived from the same `get_recent_jobs_by_company()` call which only fetches jobs from the "last week" (parameter: `date_posted: "week"`). So `jobs_30d` will actually represent at most the last 7 days' count. The 30-day average will be computed as `jobs_7d / 30`, which is dramatically lower than a true 30-day average would be. This means almost any current week activity will appear as a spike against this artificially low baseline.

**Assessment:** FALSE SPIKE DETECTION. The `jobs_30d` metric is wrong — it reflects 7-day data divided by 30, systematically generating false hiring blitz signals.

**Recommendation:** Use two separate API calls with different `date_posted` parameters to get true 7d and 30d counts. Or correct the baseline to be `daily_avg = signal.jobs_7d / 7`.

---

**Parse failure fallback: `signal.jobs_7d += 1`** (line 283)

```python
except Exception:
    signal.jobs_7d += 1  # Assume within week if can't parse
```

If the job API returns jobs with malformed or missing timestamps (which RapidAPI endpoints frequently do), every unparseable job increments the 7-day count. A bad API response with 50 malformed entries would be counted as 50 recent hires.

**Assessment:** DATA CORRUPTION ON BAD INPUT. Any timestamp parsing failure inflates the hiring count.

---

### 2.7 HouseStockWatcher — `mae_core/market/apis/house_stock_watcher.py`

**RapidAPI caps at 100 trades** (line 193)

```python
trades = self._get_trades_from_rapidapi(limit=100)  # API caps at 100
```

Congressional disclosures are 45 days delayed. In busy periods (e.g., end-of-year trading), more than 100 trades may be disclosed simultaneously. The 100-trade cap means recent high-volume disclosure periods will be silently truncated.

**Assessment:** DATA TRUNCATION. Not exploitable but could miss significant signals during disclosure floods.

---

**Free fallback S3/GitHub URLs are outdated** (lines 32-36)

The S3 bucket URL `house-stock-watcher-data.s3-us-west-2.amazonaws.com` and the GitHub mirror are community-maintained and may be stale or defunct. The housestockwatcher.com API endpoint has had reliability issues. The fallback chain provides false resilience — all three may be returning stale data or 404 responses, but the error handling would just try each and move on.

**Assessment:** SILENT DATA STALENESS. The free fallbacks may return months-old data without any indication.

---

## SECTION 3: Integration Hazards (Standalone to Live)

### 3.1 All Qdrant URLs are hardcoded localhost

`cluster_detector.py` line 21, `filing_time_analyzer.py` line 102, `contract_predictor.py` line 32.

When wired into Mae's bootstrap, these must use the existing `ApiGateway` and `BoundaryMembrane` infrastructure. Direct `requests.post()` to localhost bypasses:
- Connection Registry (no triadic witnessing of these connections)
- BoundaryMembrane rate limiting and trust scoring
- HolonRegistry awareness
- Error propagation through the organism

**Assessment:** ALL QDRANT CALLS ARE ARCHITECTURAL VIOLATIONS that must be refactored before bootstrap integration.

---

### 3.2 All modules use `print()` for logging

Every error condition and status message uses `print()`. When running inside Mae's bootstrap with multiple agents, these prints will interleave unreadably and will not be captured by any structured logging system. The convergence alerter's state is entirely in memory with no EventBus publication.

**Assessment:** NO OBSERVABILITY. No way to monitor, trace, or alert on the market modules without adding proper logging.

---

### 3.3 Thread Safety: None

`ThompsonSampler._save_distributions()` does a full JSON file write on every `update()` call. With multiple agents updating simultaneously, file writes will interleave and corrupt `thompson_distributions.json`. The file is read-parse-modify-write with no locking.

`ConvergenceAlerter.signals` dict and `alerts` list have no locking.

`learning_config.py`'s `LEARNING_CONFIG` module-level dict has no locking.

**Assessment:** DATA RACE CONDITIONS for any multi-agent execution. First integration with Mae's multi-agent system will corrupt the Bayesian state.

---

### 3.4 Circular Import Risk

`politician_tracker.py` imports from both `sec_edgar` and `usa_spending`. `contract_predictor.py` imports from `job_tracker`, `sam_gov`, and `sec_edgar`. If Mae's bootstrap Layer 33 attempts to import and initialize all market modules together, import order dependency must be respected. No circular imports currently exist, but the initialization of `SECEdgarClient()` and `USASpendingClient()` in `PoliticianTracker.__init__()` and `ContractPredictor.__init__()` creates live HTTP clients at construction time. If these constructors are called during bootstrap before network availability is confirmed, silent failures begin immediately.

---

### 3.5 No Standard Signal Format

When edge detectors produce signals, there is no common output format that the ConvergenceAlerter can directly consume. `ClusterSignal`, `FilingTimeSignal`, `HiringSignal`, `ContractPrediction`, `CorrelationSignal` — each is a different dataclass with different fields. The ConvergenceAlerter expects `record_signal(signal_id, strength, domain, direction, confidence, velocity)`. No adapter layer exists. Wiring them together requires manual translation code for each signal type.

**Assessment:** INTEGRATION WORK REQUIRED. This is expected but needs an explicit adapter pattern, not ad-hoc glue in the bootstrap layer.

---

## SECTION 4: Cold Start Problem

### 4.1 Thompson Sampler — First Boot

On first boot with no `thompson_distributions.json`, the sampler calls `_seed_from_reliability()`. This creates distributions for 12 signal types from `learning_config.py`. However, signals like `insider_cluster`, `options_flow`, `congress_trade`, `contract_award`, `technical_macd`, `insider_form4`, `technical_rsi`, `politician_sell`, `rsi`, `stochastic`, `bollinger`, `williams_r`, `cci` all appear in `thompson_distributions.json` at Beta(1,1) or Beta(5,1) or Beta(6,1) — but they are NOT in `learning_config.py`'s `source_reliability`. They must have been added manually to the JSON file at some point. These 13 signals will not be seeded on next fresh boot because their source IDs don't match any key in `source_reliability`.

**Assessment:** BOOTSTRAP INCONSISTENCY. The JSON file has diverged from the seeding logic. A fresh deployment will not reproduce the current state.

---

### 4.2 ConvergenceAlerter — No Persistence

On first boot, `self.signals` is empty. No alerts will fire until signals are recorded. If the system restarts daily (reasonable for a job), the convergence window is 48 hours but signals are not persisted to disk. All signals since last boot are lost. After restart, the system needs to re-accumulate 48 hours of signals before any convergence can be detected.

**Assessment:** STATE LOSS ON RESTART. The alerter should persist its signal history to disk on shutdown and restore on startup.

---

### 4.3 CorrelationTracker — Requires 30+ Observations

With `min_observations=10` and `window_size=30`, the correlation tracker produces no useful output until each tracked signal has 10+ aligned observations. For some signals (congressional trades, SAM.gov opportunities), this could take weeks to months of operation.

**Assessment:** KNOWN CONSTRAINT, not a bug. But callers must handle `None` returns gracefully, and they mostly do.

---

## SECTION 5: Missing Validation

### 5.1 `record_signal()` in ConvergenceAlerter — No Input Validation

`strength` and `confidence` accept any float. A caller passing `strength=1.5` or `confidence=-0.2` would corrupt convergence calculations. The threshold check `s.strength >= self.min_strength` would immediately include or exclude incorrectly.

**Recommendation:** Add `strength = max(0.0, min(1.0, strength))` and `confidence = max(0.0, min(1.0, confidence))` at the top of `record_signal()`.

---

### 5.2 `VelocityDetector.record()` — Division by Zero Risk

If `timestamp == state.last_timestamp` (two observations at identical times), `dt = 0` and the `if dt > 0` guard prevents division by zero. But the observation is also silently skipped — the velocity is not updated and no error is raised. High-frequency feeds that emit multiple events with the same timestamp will silently stall velocity computation.

**Assessment:** SILENT STALL, not a crash. But debugging will be difficult.

---

### 5.3 `ThompsonSampler._save_distributions()` — No Error Handling

```python
def _save_distributions(self) -> None:
    self.persistence_path.write_text(json.dumps(self.distributions, indent=2))
```

If the disk is full, the path is read-only, or a concurrent write corrupts the file, this will raise an exception that propagates up through `update()`. The caller has no way to handle this gracefully. Bayesian state will be lost.

**Recommendation:** Wrap in try/except and implement atomic writes (write to temp file, rename). The rename operation is atomic on POSIX filesystems.

---

### 5.4 `learning_config.update_config()` — No Validation of New Values

The meta-learner can set any value to anything. Setting `learning_rate = -5` or `decay_rates.news = 2.0` (>1.0 exponential growth) or `max_reliability = 0.1` (below min_reliability) will silently corrupt the config. The config can be set to values that make the rest of the system unstable.

**Assessment:** UNSAFE SELF-MODIFICATION. A bounds check should validate new values against allowed ranges before applying.

---

## SECTION 6: Scalability Concerns

| Component | Operation | Complexity | Risk Level |
|---|---|---|---|
| `CorrelationTracker.compute_correlation()` | Timestamp alignment | O(n^2) per pair | HIGH with 50+ signals |
| `CorrelationTracker.update_correlations()` | All pairs | O(n^2 * window^2) | HIGH with 50+ signals |
| `ConvergenceAlerter._prune_old_signals()` | Called on every insert | O(n * domains) | MEDIUM at high frequency |
| `ConvergenceAlerter.self.alerts` | Unbounded list | O(n) memory | MEDIUM over time |
| `ThompsonSampler._save_distributions()` | Full JSON write per update | O(signals) | MEDIUM at high update rate |
| `RelationshipTracker.track_insider_relationships()` | Pair enumeration | O(n^2) trades | LOW at current scale |
| `scan_all_symbols()` | Sequential API calls | O(symbols * API_latency) | LOW (expected linear) |

The O(n^2) correlation computation is the primary scalability risk. At 20 signals it is fine. At 100 signals (reasonable after full wiring), each `update_correlations()` call performs 100*99/2 = 4,950 pair computations, each with 30*30=900 inner loop iterations = 4.45 million basic operations. At 200 signals: 35 million operations. This will cause noticeable lag in a real-time system.

---

## SECTION 7: The thompson_distributions.json Audit

Current state of learned distributions:

| Signal | alpha | beta | Mean | Assessment |
|---|---|---|---|---|
| sec_edgar | 15.5 | 1.5 | 0.912 | Extremely overconfident. Alpha grew from seeded 9.5 to 15.5 — 6 "successes" logged but against what ground truth? |
| 13f_filing | 9.0 | 1.0 | 0.900 | Exactly at seeded value (prior_scale=10, reliability=0.9). Zero real observations. |
| form_4 | 9.0 | 1.0 | 0.900 | Same. |
| polygon | 9.5 | 0.5 | 0.950 | Suspiciously perfect. Beta=0.5 implies Bayesian sampler will RARELY explore this. |
| insider_cluster | 1.0 | 1.0 | 0.500 | Uninformative prior. Never seeded or updated. Will be explored heavily. |
| options_flow | 1.0 | 1.0 | 0.500 | Same. |
| congress_trade | 1.0 | 1.0 | 0.500 | Same. |
| contract_award | 1.0 | 1.0 | 0.500 | Same. |
| technical_macd | 5.0 | 1.0 | 0.833 | Manually set to optimistic. No validation. |
| technical_rsi | 6.0 | 1.0 | 0.857 | Same. |
| rsi | 1.0 | 5.0 | 0.167 | Pessimistic — rsi alone has 16% reliability? Lower than reddit (30%)? Suspicious. |
| bollinger | 1.0 | 5.0 | 0.167 | Same. |

**The split-brain problem:** `technical_macd` (mean=0.833) and `technical_rsi` (mean=0.857) are in one block, while `rsi` (mean=0.167) and `bollinger` (mean=0.167) are in another. These appear to be duplicate signal categories with contradictory reliability beliefs. The sampler will never use `rsi` or `bollinger` but will always prefer `technical_rsi` and `technical_macd` — even though these are likely measuring the same underlying signal quality.

**Assessment:** SIGNAL DUPLICATION CONFUSION. The JSON has been manually edited at some point, creating duplicate entries with contradictory values. The seeding code only creates entries for the 12 keys in `source_reliability` in `learning_config.py`, but the JSON has 22 entries. The extras were added outside the seeding system and have no path back to the config.

---

## SECTION 8: Specific Tuning Recommendations

| Parameter | Current Value | File | Recommended | Reason |
|---|---|---|---|---|
| `DEFAULT_PRIOR_SCALE` | 10 | thompson_sampler.py:32 | 2 | Reduces prior overconfidence before real observations |
| `min_variance` for get_uncertain_signals | 0.01 | thompson_sampler.py:311 | 0.001 | Current threshold excludes all seeded signals from exploration |
| `min_domains` | 2 | convergence_alerter.py:89 | 3 | Requires genuine triadic confirmation |
| `convergence_window_hours` | 48 | convergence_alerter.py:93 | 72 | 48h window too tight for slow-moving signals (insider buys can cluster over a week) |
| `anomaly_threshold` (velocity) | 2.5 | velocity_detector.py:72 | 3.0 | Financial data is non-Gaussian; 2.5 generates excessive false positives |
| `min_observations` (correlation) | 10 | correlation_tracker.py:60 | 30 | 10 observations is statistically inadequate for reliable Pearson |
| `window_size` (correlation) | 30 | correlation_tracker.py:56 | 60 | 30 days barely covers one market regime |
| `spike_ratio` threshold | 2.0 | job_tracker.py:307 | 3.0 | Current baseline is artificially low (7-day data divided by 30) |
| `correlation_threshold` for leading pairs | 0.7 | correlation_tracker.py:329 | 0.75 | With small samples, 0.7 is too easily reached by chance |
| `min_observations` (velocity) | 5 (detect_velocity_anomalies) | velocity_detector.py:195 | 10 | 5 is too few for stable z-score estimation |
| Timing window for politician-contract | -30 to +90 days | politician_tracker.py:251 | -60 to +0 days | Only predictive trades (before contract) matter |
| `is_active_bidder` default | True | contract_predictor.py:189 | False | Don't award +0.20 confidence without verification |

---

## Summary of Critical Breaks (Priority Order)

1. **`trade.is_purchase` AttributeError** — `contract_predictor.py:232` — GUARANTEED CRASH on first real data.

2. **No forgetting in Thompson Sampler** — `thompson_sampler.py:197-254` — Learned state degrades as signals change but the sampler never forgets. Early noise locked in permanently.

3. **`jobs_30d` calculated from 7-day API data** — `job_tracker.py:302-307` — Systematically generates false hiring blitz signals.

4. **Qdrant point IDs use `hash()` which is non-deterministic** — `cluster_detector.py:620` — Duplicate signals accumulate on every process restart.

5. **Thread-unsafe file writes** — `thompson_sampler.py:129-130` — Multi-agent operation will corrupt `thompson_distributions.json`.

6. **No timezone handling in FilingTimeAnalyzer** — `filing_time_analyzer.py:125-135` — Filing time patterns systematically misclassified for UTC timestamps.

7. **Velocity units in per-second** — `velocity_detector.py:133-135` — ConvergenceAlerter urgency will always read "days" for daily-frequency signals.

8. **`min_domains=2` default** — `convergence_alerter.py:89` — Alerts on single-company correlated events rather than true cross-domain convergence.

9. **`KNOWN_POLITICIANS` has 4 entries** — `politician_tracker.py:97` — The entire politician edge detector is functionally inoperable for the other 535 members of Congress.

10. **Alert duplication with no deduplication** — `convergence_alerter.py:200-218` — Will flood downstream consumers when wired to EventBus with periodic polling.
