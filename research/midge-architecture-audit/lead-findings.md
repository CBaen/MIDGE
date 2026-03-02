# MIDGE Signal Pipeline Audit — Lead Findings
**Lens: Signal Pipeline Architect**
**Date: 2026-03-01**
**Auditor role: Trace data flow from raw ingestion to tradeable output. Identify where edge exists, where it leaks, and what is actually broken.**

---

## 1. Executive Summary

MIDGE has sophisticated infrastructure for signal collection, Bayesian learning, and hypothesis generation, but a hard break exists between the pattern-detection layer and any executable trading output. The convergence alerter produces structured alerts; those alerts update an in-memory advisory dict; agents read the advisory dict and deposit pheromone markers — and then nothing happens. There is no trade execution path, no position sizing integration at the output stage, and no external notification or order routing of any kind. The system is a complete signal research and learning organism that currently has no mechanism to turn a high-confidence convergence alert into a trade. Separately, the Thompson Sampler holds meaningful learned distributions for only three sources — finra_short (1,265 observations), finnhub_earnings (71 observations), and sec_form4 (11 observations) — while the majority of sources remain at uninformative Beta(1,1) priors. The discovery log reveals a deduplication failure that logged 20+ identical alerts in the same second during a test run, indicating a suppression mechanism that works in unit tests but failed under actual marathon conditions. These three issues — no output path, thin Thompson data on most sources, and alert storm deduplication failure — are the primary financial impact concerns.

---

## 2. Signal Flow Map

The actual path from market event to system output, traced file by file:

```
MARKET EVENT (e.g., SEC Form 4 filing)
    |
    v
[sensing_fetchers.py: fetch_sec_form4()]
  - Calls get_recent_form4s(ticker, days=30)
  - Iterates watchlist tickers
  - Converts InsiderTrade -> MarketSignal via from_insider_trade()
  - No timestamp-of-event capture; uses filing date
    |
    v
[sensing_lifecycle.py: enrich_signal()]  <-- runs in background thread
  - VelocityDetector.record() -> sig.velocity
  - FilingTimeAnalyzer.analyze_filing_time() -> confidence modifier (+/-)
  - Ollama form8k_sentiment.classify() -> direction override (if configured)
    |
    v
[sensing_hook.py: _collect_one()] <-- returns to MAIN THREAD
  - Loops signals, calls convergence_alerter.record_signal(**sig_kwargs)
  - Routes to tier alerter (tactical/strategic/thematic) by TIER_ROUTING
  - Publishes CH_SIGNAL_INGESTED to EventBus (per signal)
  - Registers signals with outcome_collector
  - Stores to Qdrant + data/midge/signals/YYYY-MM-DD.jsonl
    |
    v
[convergence_alerter.py: record_signal()]
  - Appends to self.signals[domain] deque
  - Prunes signals outside 72-hour window
    |
    v
[convergence_alerter.py: check_convergence()]  <-- called from step hook
  - _check_direction_convergence("bullish") / ("bearish")
  - Requires min_domains=3 matching signals above min_strength=0.6
  - Computes Thompson-weighted geometric mean confidence
  - Checks 4-hour deduplication interval
  - Returns ConvergenceAlert objects
  - If cross_domain_count >= 3: appends to discovery_log.jsonl
    |
    v
[market_actions.py: _convergence_deepen()] <-- for MARKET_ANALYST role only
  - Reads alert dict from agent._market_advisory_ref
  - Reads kelly from ctx._latest_kelly (if available)
  - Calls agent.deposit_marker("DISCOVERY", intensity=strength, metadata={...})
  - Returns reward float 0.0-0.5 to agent lifecycle
    |
    v
*** OUTPUT GAP — NO FURTHER PROCESSING ***
  - Alert stored in convergence_alerter.alerts (capped at 500)
  - Alert written to discovery_log.jsonl
  - Pheromone marker deposited in mycelium (MycelialAgent substrate)
  - Kelly sizing computed and stored in ctx._latest_kelly
  - data/midge/convergence_state.json overwritten every 100 steps (heartbeat)
  - agent_activity.jsonl + hypothesis_activity.jsonl written
  - NO: order submission
  - NO: webhook/notification
  - NO: paper trading simulation
  - NO: external signal to any downstream system
```

**Latency estimate:** A market event (e.g., Form 4 filed 2026-02-27) reaches MIDGE as follows:
- SEC EDGAR posts filings with up to 2-day delay (form states days=30 lookback)
- fetch_sec_form4 runs every 50 steps * fetch_cadence = on source rotation (19 sources, 3 concurrent, ~50-step cadence = effectively every ~315 steps minimum before this source gets a slot)
- Enrichment + convergence check: sub-second
- Total from filing date: **2 days + rotation lag (minutes to hours depending on step speed)**

For session sweeps (ICT): near-real-time (yfinance 1-min data, kill-zone guard active), but only fetched when inside kill zone windows, and only for ES=F and NQ=F.

---

## 3. Edge Analysis

### Where Edge Exists

**3.1 Session Sweep + IFVG (strongest documented edge)**
Backtest results in `data/market/sweep_backtest_results.json` (referenced in MEMORY.md): 382 filtered trades at quality >= 0.40 show 44.4% win rate, +0.402R expectancy, PF 1.84. Elite tier (>0.65 quality) at 45.3% WR. These are real historical outcomes on 25 symbols over 48 days of 1-minute candles. Thompson keys `sweep_bt:CL=F` (alpha=14.88, beta=13.52, mean=0.524, ~27 obs) and `sweep_bt:YM=F:bearish` (alpha=8.70, beta=6.11, mean=0.588, ~13 obs) and `sweep_bt:CL=F:bearish` (alpha=8.70, beta=6.11, mean=0.588, ~13 obs) show learned positive edge above 0.5. This is the most financially grounded signal in MIDGE.

**3.2 FINRA Short Interest (most trained distribution)**
`finra_short` default: alpha=452.65, beta=812.67, mean=0.358, ~1,263 observations. Sideways: alpha=797.09, beta=1229.79, mean=0.393, ~2,025 observations. MEMORY.md cites 35.8% win rate at 5% threshold from the accelerate_learning.py run (1,987 samples). This is the most data-rich distribution, but 35.8% win rate at a 5% threshold with a direction assumption is marginal edge — barely above random if the 5% move happens in the wrong direction with similar frequency.

**3.3 SEC Form 4 (meaningful but trending negative)**
`sec_form4` default: alpha=4.57, beta=8.13, mean=0.360, ~11 observations. Sideways: alpha=4.09, beta=2.98, mean=0.578, ~5 observations. MEMORY.md cites 36.0% win rate at 5% threshold from 18 samples. The sideways regime distribution (mean=0.578) is promising but based on only 5 observations — statistically insufficient. The default-regime distribution (mean=0.360) is below chance for a binary prediction at 5% threshold. The predictions.jsonl shows a critical data quality problem: `sec_form4:GOOGL:01/25/2027` — a timestamp one full year in the future (line 21 of predictions.jsonl), which will never resolve and skews the pending prediction pool.

**3.4 Finnhub Earnings (positive but thin)**
`finnhub_earnings` default: alpha=19.07, beta=52.50, mean=0.266, ~70 observations. Sideways: alpha=28.77, beta=52.05, mean=0.356, ~80 observations. MEMORY.md cites 30.9% win rate from 110 samples. 30.9% win rate on a binary directional prediction at 5% threshold represents negative expected value unless the system is measuring magnitude-weighted returns (which outcome_collector.py does not do — it counts successes as binary).

### Where Edge Leaks

**3.5 Rotation dilution:** 19 sources rotate through a 3-slot concurrent ThreadPoolExecutor. In a 50-step cadence, each source is eligible every ~315 steps (19/3 × 50). During an active opportunity window, the sensing hook may not re-query the relevant source for hundreds of steps. Session sweeps have a kill-zone guard that skips outside active windows — this is correct — but the rotation means that even within a kill zone, the sweep source may not be polled for several minutes of wall-clock time.

**3.6 72-hour convergence window washes out slow signals:** Congressional trades (disclosure lag up to 45 days), COT data (weekly, 3-day publication lag), and SAM.gov (months-long pipelines) are all fed into a 72-hour convergence window. A COT signal recorded on Monday is gone from the convergence buffer by Thursday. These signals have no persistent weight in the convergence engine — they can only contribute if they happen to be in the 72-hour window when a convergence check fires.

**3.7 min_domains=3 blocks high-conviction single-domain alerts:** The session sweep + IFVG pattern is a self-contained technical signal with documented edge. It fires in the `technical` domain. Unless two other domains independently contribute bullish/bearish signals within 72 hours on any ticker, the sweep signal never generates a convergence alert. This is mathematically sound (Law 2) but operationally means MIDGE's best-documented edge source is blocked from generating actionable output unless unrelated signals happen to coincide.

**3.8 Confidence score disconnect from actual accuracy:** The `_compute_confidence()` method in convergence_alerter.py computes a Thompson-weighted geometric mean of signal confidences. But signal confidences are set by the adapter functions at a static level (e.g., from_insider_trade sets confidence=0.70 by default; from_congressional_trade is 0.75). These static priors are not re-calibrated against the actual observed outcomes in outcomes.jsonl. The Thompson weights adjust how much each source influences the geometric mean, but do not change the base confidence values. A source with mean=0.36 in Thompson still contributes its 0.70 static confidence to the formula, just down-weighted.

---

## 4. Critical Issues

### Issue 1: No Trade Execution Path (Severity: Fatal for financial utility)
**Location:** Entire output layer — no single file responsible.

After a convergence alert is generated in `convergence_alerter.py:check_convergence()`, the alert propagates to:
- `market_actions.py:_convergence_deepen()` — deposits a pheromone marker
- `data/midge/convergence_state.json` — heartbeat file
- `data/midge/agent_activity.jsonl` — activity log

None of these outputs leave the MIDGE process boundary. There is no webhook, no broker API call, no paper trading simulation, no email/SMS notification, no file written to a shared location for consumption by another process. The `TradeSignal` dataclass is defined in `signal.py` (lines 82-94) but is never instantiated anywhere in the codebase. MIDGE can detect patterns with high confidence and produce zero tradeable output. This is the critical gap between "organism that learns" and "organism that trades."

### Issue 2: Discovery Log Alert Storm (Severity: High — corrupts learning signal)
**Location:** `data/market/discovery_log.jsonl` lines 10-30+, and `convergence_alerter.py:check_convergence()`

The 4-hour deduplication interval (`_min_alert_interval_hours = 4.0`) at line 201 of `convergence_alerter.py` works correctly in isolation. However, the discovery log shows 20+ identical alerts logged within a single second on 2026-02-27 18:33:42:

```
CONV-20260227-0001 through CONV-20260227-0021 — all same sources, same strength=0.916
```

This means `_log_discovery()` is being called repeatedly without the deduplication guard being consulted. The guard in `check_convergence()` (lines 424-437) updates `_last_alert_time` on first alert, but if `check_convergence()` is called in a tight loop (e.g., from HolonProxy.step() AND the sensing hook AND agent steps simultaneously within the same wall-clock second), the same alert fires 20 times. Each logged discovery enters the discovery_log used for RSI Layer 2 pattern seeding. This is corrupting the discovery signal — the system thinks it found 20 novel patterns when it found one.

Root cause: The deduplication guard compares wall-clock seconds (`datetime.now()`), which are non-monotonic in fast loops. Multiple calls within the same second all pass the check because `_last_alert_time` is only updated after appending to `filtered` — and concurrent calls race on that update.

### Issue 3: Prediction Data Integrity — 1-Year Future Timestamp (Severity: High)
**Location:** `data/market/predictions.jsonl` line 21

```json
{"signal_id": "ebd0b768...", "source": "sec_form4", "symbol": "GOOGL",
 "direction": "down", "timestamp": "2026-02-22T20:08:00.199189",
 "outcome_window_days": 45, "outcome_symbol": "GOOGL",
 "metadata": {"original_signal_id": "sec_form4:GOOGL:01/25/2027"}}
```

The raw_id `sec_form4:GOOGL:01/25/2027` contains a year 2027 date being processed as a 2026 prediction. `_ensure_datetime()` in `signal.py` (lines 24-36) tries multiple format parsers and falls back to `datetime.now()` on failure — but here it apparently succeeded and parsed a real 2027 date. The OutcomeCollector registered this as a prediction with a 45-day outcome window anchored to Jan 25, 2027. This prediction will never mature (it's 11 months in the future), and it is sitting in the `registered_signals.json` preventing re-registration of any corrected version. The sec_form4 Thompson distribution (11 observations) is contaminated by this and potentially similar artifacts.

### Issue 4: Thompson Prior Mismatch — learning_config vs actual distributions (Severity: Medium)
**Location:** `data/market/thompson_distributions.json` vs `learning_config.py` source_reliability

The `learning_config.py` lists `source_reliability` for `congressional = 0.75`, which seeds Beta(1.5, 0.5) — a prior mean of 0.75. The actual `thompson_distributions.json` shows:

```json
"congressional": {"default": {"alpha": 5.72, "beta": 29.21}}
```

Mean = 5.72 / (5.72 + 29.21) = **0.164** — dramatically below the 0.75 configured prior. This is a MEMORY.md-documented result (16.4% win rate from 53 samples). The learning_config default is dangerously optimistic compared to what the data shows. When ThompsonCalibrator runs Wire 1 of meta-learning (hypothesis_engine.py line 733-776), it reads the calibration feedback and may attempt to raise the `source_reliability.congressional` key — but the actual Thompson distribution is already correctly capturing the low mean. This creates a disconnect: the config says 0.75 is the baseline, the distribution says 0.164 is the reality. Any new session that seeds from `learning_config` without a saved `thompson_distributions.json` will dramatically overestimate congressional reliability for the first N steps until the distribution is loaded.

The identical mismatch exists for `sec_edgar = 0.95` in config, `sec_edgar` distribution: alpha=1.48, beta=1.0, mean=0.597 (minimal observations, not meaningfully calibrated).

### Issue 5: Outcome Evaluation — Repeated prediction re-registration (Severity: Medium)
**Location:** `data/market/outcomes.jsonl` lines 1-10; `outcome_collector.py`

The outcomes.jsonl shows the same 4 prediction IDs being evaluated multiple times:
- `prediction_id: 5f6b90d7-41ca-425a-8553-a6d1136ac89f` (AAPL bullish) appears 3 times with different outcome prices
- The first evaluation (line 1) shows outcome=189.25, return=2.02%
- The third evaluation (line 7) shows outcome=278.12, return=49.93%

This is not the same prediction being resolved at different times — the entry_price is 185.5 in both, but the outcome price jumped from 189.25 to 278.12 between evaluations. The 49.93% return is not a real 1-day return for AAPL. These appear to be synthetic test records from the initial bootstrap, but they are now mixed with real outcome records and are updating the Thompson distributions. Mock data is poisoning the real Bayesian learning loop.

### Issue 6: SEC Form 4 Signal Direction Bias — Systematic sell-side skew (Severity: Medium)
**Location:** `data/market/predictions.jsonl` lines 11-30

Every recent sec_form4 prediction in the predictions file has `"direction": "down"` — MSFT, GOOGL (6 entries), LIFE, AMZN (3 entries), NVDA (2 entries), META (3 entries). The only "up" signal is one MSFT entry. This reflects that tech executives in early 2026 were predominantly selling (RSU vesting, planned sales). The system is correctly reading the data, but the convergence engine will struggle to generate bullish convergence from sec_form4 signals during sustained sell periods. MIDGE's watchlist (AAPL, MSFT, GOOGL, AMZN, NVDA, META, LMT, RTX, NOC, GD, BA) skews toward tech mega-caps where insider selling is routine and uninformative. The cluster_detector.py threshold of 3+ insiders buying would be a stronger filter than raw form4 direction for this watchlist.

---

## 5. Improvement Opportunities (ranked by expected financial impact)

### Rank 1: Build the Output Path — TradeSignal → Broker/Notification
**Expected impact: Required for any financial return**

The `TradeSignal` dataclass in `signal.py` is defined but never instantiated. A minimal viable output layer would:
1. Convert ConvergenceAlert → TradeSignal when confidence > threshold
2. Apply Kelly sizing (already computed in `ctx._latest_kelly`)
3. Write to a watched output file or call a webhook

This does not require broker integration at first — a paper trading simulation that writes to `data/midge/paper_trades.jsonl` with entry price, direction, and kelly-sized notional is sufficient to verify the pipeline end-to-end and start accumulating real P&L data.

The entry point is clear: in `sensing_hook.py:_collect_one()`, after `check_convergence()` returns alerts, the existing code feeds the advisory dict. Add: if alert.confidence > threshold, instantiate TradeSignal and write it. The signal router and Kelly components already exist.

### Rank 2: Unlock Session Sweep Signals from min_domains=3 Requirement
**Expected impact: Makes MIDGE's best-documented edge (PF 1.84) actionable without requiring coincidental multi-domain agreement**

The sweep + IFVG pattern has documented edge from backtesting. The min_domains=3 requirement is a triadic law compliance mechanism (Law 2), not a statistical necessity for this specific signal type.

Two approaches:
- Option A: Add a `fast_track` flag to ConvergenceAlert for BACKTEST_DERIVED patterns with DSR > threshold, bypassing the domain count requirement
- Option B: Create a parallel direct-output path for session_sweep_ifvg signals that bypasses the convergence engine entirely, writing TradeSignal directly when quality >= 0.65 (Elite tier)

Option B preserves Law 2 for the convergence engine while allowing the explicitly backtested pattern to fire independently.

### Rank 3: Fix Discovery Log Deduplication — Thread-Safe Alert Suppression
**Expected impact: Removes corruption from RSI Layer 2 training data**

Replace the wall-clock comparison in `convergence_alerter.py:check_convergence()` (line 430) with a monotonic clock comparison, and add a threading.Lock() around the deduplication check-and-update block. The current race condition:

```python
# CURRENT (racy):
if (now - self._last_alert_time).total_seconds() / 3600 < self._min_alert_interval_hours:
    continue
filtered.append(alert)
self._last_alert_direction = direction
self._last_alert_time = now  # <-- update happens AFTER multiple callers pass the check
```

Should be:
```python
# FIX:
with self._dedup_lock:
    if (monotonic() - self._last_alert_monotonic) < self._min_alert_interval_seconds:
        continue
    self._last_alert_direction = direction
    self._last_alert_monotonic = monotonic()
filtered.append(alert)
```

### Rank 4: Replace Mock Outcomes with Real Data, Purge Contaminated Records
**Expected impact: Cleans the Bayesian learning signal**

The outcomes.jsonl contains synthetic test records with impossible return values (49.93% in 1 day for AAPL). These update the Thompson distributions. Steps:
1. Identify all predictions with `prediction_source: "midge"` and `outcome_due` within 1 day — these are the mock records (timeframe="1d" with same-day outcome_due is the tell)
2. Remove them from outcomes.jsonl and rebuild Thompson distributions by replaying only the real outcomes (those with realistic price moves and proper timeframes)
3. Fix the 2027-timestamp prediction (sec_form4:GOOGL:01/25/2027) in registered_signals.json

### Rank 5: Fix Slow-Signal Domain Persistence — Extend Convergence Window or Add Pinning
**Expected impact: Allows COT, congressional, and SAM.gov signals to contribute to convergence**

COT data updates weekly. Congressional disclosures arrive up to 45 days late. Both sources have learning_config reliability estimates > 0.5, but both wash out of the 72-hour convergence window before they can participate in convergence. Two options:
- Extend `convergence_window_hours` for slow domains only (per-domain window map)
- Add a "pinned signal" mechanism: when a high-confidence slow signal arrives, pin it to the convergence buffer with a 14-day TTL regardless of the global window

The second option preserves the 72-hour window for fast signals (technicals, news) while giving slow signals persistence proportional to their known alpha decay rates in `learning_config.py:decay_rates`.

### Rank 6: Calibrate Static Signal Confidence to Thompson Reality
**Expected impact: Prevents misleadingly high convergence confidence scores**

The adapter functions in `signal_adapters/` set static confidence values (e.g., congressional=0.75, insider=0.70). The Thomson distributions show: congressional mean=0.164, sec_form4 mean=0.360. The confidence formula uses the static values as inputs and Thompson weights as multipliers. A congressional signal enters with confidence=0.75, gets down-weighted by Thompson (weight = 0.5 + 0.164 = 0.664), but the geometric mean still includes 0.75 as the signal confidence. The output is inflated.

Fix: replace static confidence in adapter functions with a `_resolve_confidence(source, thompson_sampler)` lookup that uses `0.5 + dist.mean` directly as the signal confidence when Thompson has >= 5 observations. This aligns the input to the confidence formula with the actual learned reliability.

### Rank 7: Watchlist Diversification — Add Instrument-Agnostic Opportunities
**Expected impact: Removes systematic tech-sell bias from Form 4 signal pool**

The default watchlist (`sensing_lifecycle.py:load_watchlist()`, lines 113-124) is: AAPL, MSFT, GOOGL, AMZN, NVDA, META, LMT, RTX, NOC, GD, BA. This is 6 mega-cap tech companies and 5 defense contractors. Mega-cap tech insiders routinely sell for estate planning, tax optimization, and diversification — none of which is informative. The cluster_detector requires 3+ insiders buying (bullish), which is rare in mega-caps. Suggestion: replace 3-4 mega-cap slots with mid-cap names where insider buying is more signal-rich, or filter Form 4 signals to exclude pre-planned 10b5-1 plans (already partially handled by `sensing_fetchers.py` but the watchlist selection amplifies the sell bias).

---

## 6. Data Evidence

### Thompson Distribution Summary (from `data/market/thompson_distributions.json`)

| Source | Alpha | Beta | Mean | Samples | Regime |
|--------|-------|------|------|---------|--------|
| finra_short | 452.65 | 812.67 | 0.358 | 1,263 | default |
| finra_short | 797.09 | 1,229.79 | 0.393 | 2,025 | sideways |
| yfinance_price | 85.04 | 285.43 | 0.230 | 368 | default |
| yfinance_price | 143.16 | 500.89 | 0.222 | 642 | sideways |
| finnhub_earnings | 19.07 | 52.50 | 0.266 | 70 | default |
| finnhub_earnings | 28.77 | 52.05 | 0.356 | 79 | sideways |
| congressional | 5.72 | 29.21 | 0.164 | 34 | default |
| sec_form4 | 4.57 | 8.13 | 0.360 | 11 | default |
| sec_form4 | 4.09 | 2.98 | 0.578 | 5 | sideways |
| contract_award | 4.24 | 23.29 | 0.154 | 26 | default |
| sweep_bt:CL=F | 14.88 | 13.52 | 0.524 | 27 | default |
| sweep_bt:YM=F:bearish | 8.70 | 6.11 | 0.588 | 13 | default |
| sweep_bt:CL=F:bearish | 8.70 | 6.11 | 0.588 | 13 | default |
| cot_positioning | 23.09 | 59.58 | 0.279 | 81 | sideways |
| All remaining sources | ~1.0 | ~1.0 | ~0.50 | 0 | any |

**Key finding:** 29 of 46 distribution keys have 0 real observations (Beta(1,1) uninformative prior). The 17 with meaningful data are concentrated in finra_short (dominant), yfinance_price, and a handful of backtest-derived sweep keys. The sweep_bt keys are the only distributions showing means consistently above 0.50 in default regime.

### Discovery Log Analysis (from `data/market/discovery_log.jsonl`)

**Early entries (2026-02-08):** Synthetic discovery cycles with mock patterns. "3/4 outcomes correct" — these are artificial bootstrap records, not real market discoveries.

**First real alert (2026-02-25):** CONV-20260225-0001, bullish, domains=[crypto, government, insider], strength=0.75, confidence=0.73. The "crypto" domain is not a configured data source in the current sensing pipeline — this alert came from a manual test or earlier version of MIDGE.

**2026-02-27 alert storm:** CONV-20260227-0001 through CONV-20260227-0021+, all logged within 1 second, all identical. Sources: `finnhub_earnings:SHO`, `social:SOFI`, `session_sweep:NQ=F:new_york:high_sweep`. This is the deduplication failure documented in Issue 2. These 20+ records in the discovery log will be treated as 20 independent pattern discoveries by RSI Layer 2 if the generator ever reads from the discovery_log (current implementation reads lag_correlations.json instead, so this is not yet causing hypothesis generation errors — but it would if a discovery-log-to-hypothesis bridge were added).

### Prediction Quality (from `data/market/predictions.jsonl`)

**Lines 1-10:** Mock predictions from 2026-02-08, all pointing to SPY bullish, generated by synthetic discovery cycles. Entry/target/stop all = 0.0. These have `outcome_recorded: false` and `outcome_due: 2026-02-09` — they expired unresolved and contributed nothing to the learning loop.

**Lines 11+:** Real OutcomeCollector-registered predictions from 2026-02-22, all from sec_form4, all directional "down" except one "up" for MSFT. These have proper 45-day windows and will mature around 2026-04-08. The sec_form4 Thompson distribution with mean=0.360 on 11 observations is partly built from these. No outcomes yet resolved for this batch.

**Structural problem:** Two schema variants coexist in predictions.jsonl — the old format (fields: prediction_id, symbol, direction, confidence, entry_price, target_price, stop_loss, reasoning, contributing_signals, predicted_at, outcome_due, timeframe, outcome_recorded, was_correct, return_pct, prediction_source) and the new format (fields: signal_id, source, symbol, direction, timestamp, outcome_window_days, outcome_symbol, metadata). The OutcomeCollector reads both, but the old-format records have `entry_price=0.0` and `target_price=0.0`, making price-based outcome evaluation meaningless for them.

### Outcomes File Analysis (from `data/market/outcomes.jsonl`)

**Lines 1-2:** Real format outcomes — AAPL bullish correct (2.02% return, 1d), MSFT bearish incorrect (1.19% wrong direction). These look real but the 1-day timeframe and exact-match timestamps suggest they are from an early bootstrap test.

**Lines 3-10:** The same 4 prediction IDs re-evaluated with different (wrong) prices. AAPL "return" of 49.93% in 1 day is physically impossible. These are the mock data contamination records documented in Issue 5.

---

## 7. Appendix: Key File References

| File | Role in Pipeline | Critical Lines |
|------|-----------------|----------------|
| `C:\Users\baenb\projects\MIDGE\mae_core\market\sensing_hook.py` | Source rotation orchestration | L104-126 (SOURCE_ROTATION), L189 (fetch_cadence=50), L246-259 (step()) |
| `C:\Users\baenb\projects\MIDGE\mae_core\market\sensing_fetchers.py` | 19 fetch functions | L29 (days=30 lookback), L293-302 (kill zone guard) |
| `C:\Users\baenb\projects\MIDGE\mae_core\market\sensing_lifecycle.py` | Enrichment | L40-50 (FilingTimeAnalyzer integration), L55-68 (Ollama integration) |
| `C:\Users\baenb\projects\MIDGE\mae_core\market\intelligence\convergence_alerter.py` | Crown jewel | L141 (min_domains=3), L201 (4h dedup), L424-437 (dedup logic — race condition) |
| `C:\Users\baenb\projects\MIDGE\mae_core\market\intelligence\thompson_sampler.py` | Bayesian weights | L379-407 (apply_forgetting), L197-209 (sample) |
| `C:\Users\baenb\projects\MIDGE\mae_core\market\intelligence\outcome_collector.py` | Feedback loop | L42-51 (OUTCOME_WINDOWS), L53 (SUCCESS_THRESHOLD_PCT=5.0) |
| `C:\Users\baenb\projects\MIDGE\mae_core\market\market_actions.py` | Agent output | L268-299 (_convergence_deepen — terminal point of alert processing) |
| `C:\Users\baenb\projects\MIDGE\mae_core\market\signal.py` | TradeSignal definition | L82-94 (defined but never instantiated) |
| `C:\Users\baenb\projects\MIDGE\data\market\thompson_distributions.json` | Learned distributions | All — see table above |
| `C:\Users\baenb\projects\MIDGE\data\market\discovery_log.jsonl` | Alert history | Lines 10-30+ (alert storm evidence) |
| `C:\Users\baenb\projects\MIDGE\data\market\predictions.jsonl` | Prediction tracking | Line 21 (2027 timestamp bug), Lines 1-10 (mock contamination) |
| `C:\Users\baenb\projects\MIDGE\data\market\outcomes.jsonl` | Outcome tracking | Lines 3-10 (impossible returns) |
