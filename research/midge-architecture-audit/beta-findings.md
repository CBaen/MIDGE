# MIDGE Architecture Audit — Beta Findings (Systems Reliability Engineer)

**Auditor Role:** Witness Beta — Systems Reliability Engineer
**Analytical Lens:** Production runtime behavior, persistence integrity, concurrency correctness, error blast radius, and operational continuity for a system making real financial decisions.
**Audit Date:** 2026-03-01
**Files Examined:** 28 source files, 8 data files

---

## 1. Executive Summary

MIDGE has strong concurrency architecture at the macro level (ThreadPoolExecutor with skip-if-busy patterns, non-blocking step loops) but contains several persistence vulnerabilities that can silently corrupt the Bayesian learning state that drives trading decisions. The most serious issue is that `_log_update()` in ThompsonSampler appends to `thompson_history.jsonl` without a lock, creating interleaved writes across multiple background threads that can produce corrupted JSONL lines — the same category of corruption that previously required a full Thompson rebuild from outcomes (per the MEMORY.md note about "marathon file-lock corruption"). Pair outcomes data already shows test contamination ("a|b", "a0|b0" etc.) in production files. The `write_text()` calls for retirement window, pair outcomes, and config snapshot are non-atomic on Windows and will produce truncated files on any crash mid-write. The system lacks log rotation, disk growth monitoring, and a recovery path for the partial-state scenario where some persistence files are present and others are corrupted.

---

## 2. Critical Failures (Will Break in Production)

### CF-1: Thompson History JSONL Write Without Lock
**File:** `C:\Users\baenb\projects\MIDGE\mae_core\market\intelligence\thompson_sampler.py`, line 272

```python
def _log_update(self, result: UpdateResult) -> None:
    """Append update to history file."""
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")
```

`_log_update()` is called from `update()` (line 263), which is invoked by the `OutcomeCollector` on the main thread AND potentially by background threads through any future convergence path. The file open/write is not protected by `self._lock`. By contrast, `_save_distributions()` (line 134) IS protected by `self._lock`. This asymmetry means concurrent calls to `update()` can interleave their JSONL writes, producing partial or concatenated lines. MEMORY.md explicitly records this already caused a full Thompson distribution rebuild from outcomes ("Marathon file-lock corruption wiped distributions to Beta(1,1). Rebuilt by replaying 9,462 deduped outcomes from outcomes.jsonl"). The bug is not fixed — only the rebuild procedure was added.

**Blast radius:** Thompson history file corrupts silently. The rebuild procedure requires replaying thousands of outcomes via yfinance, which is rate-limited. During the rebuild window, Thompson distributions revert to priors, degrading all confidence scoring for trading signals.

### CF-2: Non-Atomic write_text() on Windows for Critical State Files
**Files:**
- `learning_config.py` line 237: `snapshot_path.write_text(json.dumps(LEARNING_CONFIG, indent=2))`
- `hypothesis_engine.py` line 476: `self._retirement_window_path.write_text(json.dumps(payload, indent=2))`
- `hypothesis_generator.py` line 193: `self._pair_outcomes_path.write_text(json.dumps(serialized, indent=2))`
- `step_timer.py` line 81: `out_path.write_text(json.dumps(payload, indent=2))`

`Path.write_text()` on Windows is NOT atomic. It truncates the file, then writes. A crash, OOM kill, or power loss between truncation and write completion produces a zero-byte or partial JSON file. On next startup, `load_snapshot()` (learning_config.py line 255) calls `json.loads(snapshot_path.read_text())` which raises `JSONDecodeError`, the `except` block logs a warning and returns `False`, and the system boots with factory defaults — silently losing all meta-learned config changes.

By contrast, `_save_distributions()` in ThompsonSampler (line 134-144) correctly uses `os.replace(tmp, path)` for atomic writes. This pattern is NOT applied consistently.

**Blast radius:** Any session crash during config update wipes the meta-learned hypothesis gates and generator thresholds. Gates reset to defaults (promote_win_rate=0.52) regardless of what the meta-learner had calibrated them to. The system will not detect this regression.

### CF-3: Pair Outcomes File Contains Test Contamination in Production
**File:** `C:\Users\baenb\projects\MIDGE\data\market\pair_outcomes.json`

```json
{
  "a|b": {"promoted": 13, "retired": 21},
  "a0|b0": {"promoted": 5, "retired": 0},
  "a1|b1": {"promoted": 5, "retired": 0},
  ...
  "a6|b6": {"promoted": 5, "retired": 0}
}
```

The production pair_outcomes.json contains entries from test runs (source names "a", "b", "a0", "b0", etc. are clearly test fixtures, not real source names). These artificial entries skew the `_finding_priority()` sort in `hypothesis_generator.py` (line 298). The "a|b" pair has 21 retired outcomes — the most retired of any pair — but it maps to no real sources, so it wastes priority budget. More concerning, the test fixtures with 5 promoted / 0 retired get 0.1 priority bonuses, pushing ahead of pairs with real performance history.

**Blast radius:** Hypothesis generation ordering is corrupted by phantom test data. The system generates a hypothesis from "sec_form4→finnhub_earnings" (the only real entry) but competes with 8 ghost pairs for priority slots.

### CF-4: Retirement Window Shows Suspicious Recency Skew
**File:** `C:\Users\baenb\projects\MIDGE\data\market\retirement_window.json`

The retirement window ends with 15 consecutive "retired" entries (indices 35-49 of 49). This is the state used by Wire 2 of meta-learning to judge retirement rate. With 15/50 = 30% retirement rate, the meta-learner is currently sitting at the boundary of the "too many retirements" threshold (>70%). However, the retirement_window is seeded from registry state at cold-start (`_seed_retirement_window_from_registry()`), meaning these "retired" entries may reflect historical state rather than live session performance. The meta-learner cannot distinguish seeded historical entries from live session ones.

**Blast radius:** Wire 2 may tighten `min_correlation` incorrectly based on seeded historical data, suppressing hypothesis generation in sessions where live performance is actually improving.

---

## 3. Race Conditions and Thread Safety

### RC-1: ConvergenceAlerter Called From Multiple Threads
**Files:** `sensing_hook.py` line 231 (ThreadPoolExecutor max_workers=3), `convergence_alerter.py` line 250 (record_signal → self.signals[domain].append)

`MarketSensingHook` uses `ThreadPoolExecutor(max_workers=3)`. Each of the 3 concurrent worker threads calls `_fetch_source()`, which enriches signals in the background (sensing_lifecycle.py line 24). Enriched signals are then collected in `_collect_results()` on the main thread via `_collect_one()`. Signal feeding to `_convergence_alerter.record_signal()` happens in `_collect_one()` (sensing_hook.py line 391) which runs on the main thread. This is safe.

HOWEVER, `convergence_alerter.step()` is called by the HolonProxy step hook (wired in bootstrap), which runs on the main thread at the same time `_collect_one()` is also running on the main thread. Both call `_prune_old_signals()` and mutate `self.signals`. Since Mesa runs step hooks sequentially this is safe in a single-threaded step loop. But if any future integration runs step hooks in parallel threads, this becomes a data race.

The real risk is `_log_discovery()` in convergence_alerter.py (line 748):
```python
with open(_DISCOVERY_LOG, "a") as f:
    f.write(json.dumps(record) + "\n")
```
This is called from `check_convergence()` which could theoretically be called from background threads. Currently safe because collection is main-thread-only, but fragile.

### RC-2: Hypothesis Engine Background Validation + Main Thread Promote/Retire
**File:** `hypothesis_engine.py` lines 290-306 (`_launch_validation`) and 356 (`_promote`)

The background validation thread (`_run_validation`, runs in `self._validation_executor`) calls `self._registry.update_stats()`, `self._promote()`, and `self._retire()`. Meanwhile, the main thread calls `request_validation()` (agent-triggered, line 168) which also calls `self._promote()` and `self._retire()`. Both paths mutate `self._hypotheses` in `HypothesisRegistry` (line 87: `self._hypotheses[hypothesis_id] = hypothesis`) without any lock.

If background validation is running and an agent simultaneously triggers `request_validation()`, two threads can call `self._registry.promote()` on the same hypothesis. The registry's `promote()` checks status first (line 76: `if hyp is None or hyp.status != HypothesisStatus.PROBATION`), but there is no lock between the check and the mutation — a classic TOCTOU (Time of Check to Time of Use) race.

**Blast radius:** A hypothesis could be double-promoted, causing a double Thompson distribution write and double retirement_window append. The Thompson write is protected by `self._lock` in `_save_distributions()`, but the retirement window `_save_retirement_window()` is not locked, so two threads could corrupt it simultaneously.

### RC-3: Thompson Sampler update() — Distributions Dict Modified Without Lock
**File:** `thompson_sampler.py` lines 240-244

```python
# Store updated distribution (NOT inside _lock)
self.distributions[signal_id][regime] = {
    "alpha": new_alpha,
    "beta": new_beta
}
# Then later: _save_distributions() acquires _lock
```

The dict mutation at line 240-244 happens BEFORE `_save_distributions()` acquires `self._lock`. The lock only protects the file write, not the in-memory dict mutation. If two threads call `update()` simultaneously for different signal_ids, the GIL provides some protection in CPython, but for the same signal_id + regime combination, the second thread can overwrite the first's update with stale values.

`get_distribution()` (line 171) also mutates `self.distributions` without a lock (it initializes new keys), which can race with `update()`.

### RC-4: JSONL Appends Are Not Atomic
**Files:** `hypothesis_registry.py` line 186, `sensing_lifecycle.py` line 85, `convergence_alerter.py` line 763, `thompson_sampler.py` line 272

All JSONL appends use `open(path, "a")` + `f.write(json.dumps(event) + "\n")`. On Windows NTFS, file append operations are NOT guaranteed atomic across processes or threads. A partial write (e.g., only half the JSON was written when the process was killed) produces a line that `json.loads()` cannot parse. The hypothesis registry's `_load_events()` (line 202) handles this with `except (json.JSONDecodeError, Exception): continue` — it silently skips corrupted events. This is graceful but means a crash during a PROMOTED event write permanently loses that promotion from the audit log.

---

## 4. Persistence Vulnerabilities

### PV-1: Config Snapshot Forward-Compatibility Gap
**File:** `learning_config.py` lines 264-276 (`_deep_merge`)

`_deep_merge` only updates keys that already exist in base: "Keys in update that are absent from base are ignored." This means if a snapshot was saved with an older code version that lacked `hypothesis_gates` or `generator_thresholds`, those new keys will never be restored from a snapshot. The warm-start will silently use factory defaults for the new keys even though a snapshot exists.

Currently this is working correctly (meta-learner has modified config version 3 as of 2026-02-28). But if a developer adds new keys to `LEARNING_CONFIG` without a migration path, future sessions silently lose the historical tuning on those keys.

### PV-2: Hypothesis Registry Has No Compaction
**File:** `hypothesis_registry.py`, `data/market/hypotheses.jsonl`

The event-sourced registry appends an event for every status change: CREATED, UPDATED (per validation cycle), PROMOTED, RETIRED, HIBERNATED, REACTIVATED. For 30+ active hypotheses with validation every 1000 steps and 5000+ steps per session, this generates 30 UPDATED events per validation cycle. At 5 validations per session: 150 UPDATED events. Over weeks of continuous operation, `hypotheses.jsonl` grows unboundedly with no compaction or rotation.

The file is fully replayed on every startup (`_load_events()`, line 191). As the file grows (potentially to tens of MB), startup time increases linearly. A 100MB file takes several seconds to replay even on fast hardware.

Additionally, `_load_events()` only keeps the LAST event for each hypothesis_id (line 204: `self._hypotheses[hypothesis_id] = hyp` overwrites on every event). Earlier events for the same hypothesis are therefore wasted disk space.

### PV-3: Registered Signals Set Grows Unbounded
**File:** `outcome_collector.py` line 86-87, line 211-215

```python
self._registered: set = self._load_registered()
# ...
self._registered_path.write_text(json.dumps(sorted(self._registered), indent=0))
```

`self._registered` is a Python set of signal IDs (strings). Every registered signal is added permanently. Over months of operation with signals arriving every few steps, this set could grow to hundreds of thousands of entries. `write_text()` then serializes the entire sorted set to JSON on every registration batch. This operation is O(n log n) for sorting and O(n) for serialization — on a large set, this can take hundreds of milliseconds, blocking the main thread for every batch.

The set is never pruned. Signal IDs are UUID-like strings with format `{source}-{symbol}-{timestamp}`, so older predictions that have long since resolved are retained forever.

### PV-4: Convergence Alerter Signals Grow Unbounded Within 72h Window
**File:** `convergence_alerter.py`, line 169-171

```python
self.signals: Dict[str, List[Signal]] = defaultdict(list)
```

`_prune_old_signals()` only removes signals older than 72 hours. With 19 sources running every 50 steps at typical simulation speeds, a 72-hour window can accumulate thousands of Signal objects across many domains. Each Signal object holds a metadata dict. The `self.alerts` list is capped at 500 (line 441), but `self.signals` has no cap. Memory grows monotonically within each 72-hour window.

---

## 5. Error Handling Gaps

### EG-1: yfinance Rate Limiting — No Backoff or Circuit Breaker
**File:** `price_fetcher.py` lines 272-301 (`_fetch_yfinance`)

When yfinance returns an empty result or raises an exception, the code logs a warning and returns `None`. There is no exponential backoff, no circuit breaker, and no rate-limiting awareness. During the `accelerate_learning.py` pipeline (noted in MEMORY.md: "~3,382 predictions pending (yfinance rate limit)"), yfinance silently rate-limits without any detection or backoff. The system keeps retrying on the same cadence, burning the rate limit without productive work.

`get_daily_history()` (line 171) is called for every ticker in the watchlist every time `ta_indicators` is fetched. With 11 tickers and a 50-step fetch cadence, this results in 11 yfinance calls per fetch cycle. During extended runs this reliably hits yfinance's undocumented rate limit.

### EG-2: Backtest Scheduler — 5 Minute Blocking Operation in "Background" Thread
**File:** `backtest_scheduler.py` line 157

```python
bt = SweepBacktester(interval="5m", days=59)
trades = bt.run(self._symbols)  # documented: 3-5 min runtime
```

The backtest runs 6 symbols × 59 days × 1-minute OHLCV data = substantial yfinance volume. MEMORY.md notes "3-5 min runtime." During this window, the background ThreadPoolExecutor(1) is blocked. If the yfinance call hangs (e.g., network timeout with no timeout parameter set), the thread is blocked indefinitely. The main thread continues stepping normally, but any subsequent call to `check_and_schedule()` sees `_pending_future.done() == False` and skips. The backtest effectively locks up silently.

There is no timeout enforcement on the backtest run. There is no watchdog to kill a hung backtest thread.

### EG-3: Session Round Exception Swallowing
**File:** `main.py` lines 588-594

```python
try:
    model.run(num_steps)
except KeyboardInterrupt:
    logger.info("Interrupted by user at round %d", r)
    interrupted = True
except Exception:
    logger.exception("Round %d failed — continuing to next round", r)
```

A round exception is logged and the loop continues. This is intentional fault tolerance. However, if the exception is caused by a corrupted hypothesis registry or a broken Thompson distribution (e.g., after an interrupted write), every subsequent round will also fail with the same exception, producing an infinite loop of failed rounds that each consume a new `run_service.bat` restart cycle. There is no failing-fast mechanism to distinguish "transient error, retry" from "persistent state corruption, requires manual intervention."

### EG-4: `run_service.bat` Does Not Handle Python Environment Failures
**File:** `run_service.bat`

```bat
:loop
python main.py --continuous --agents 5 --steps 2000
echo [%date% %time%] MIDGE exited (code %ERRORLEVEL%). Restarting in 30 seconds...
timeout /t 30 /nobreak >nul
goto loop
```

The bat file restarts after any exit, including `ImportError` (missing package), `ModuleNotFoundError` (corrupted venv), or `MemoryError`. A Python import failure exits immediately, the bat waits 30 seconds, and retries — producing an infinite crash-restart loop at 30-second intervals that fills logs and burns any logging disk budget without making progress.

There is no `ERRORLEVEL` check that would stop the loop on persistent errors. There is no maximum restart count.

### EG-5: OutcomeCollector `_save_registered()` Uses write_text — Loses All Registered IDs on Crash
**File:** `outcome_collector.py` line 214

```python
self._registered_path.write_text(json.dumps(sorted(self._registered), indent=0))
```

If the process crashes mid-write, the registered_signals.json is left empty or truncated. On restart, `_load_registered()` returns an empty set, and every previously-registered signal ID is treated as new. The `OutcomeTracker` then re-registers all these signals as fresh predictions with their original timestamps, creating duplicate predictions in `predictions.jsonl` — some of which may have already been resolved. Thompson distributions would receive duplicate outcome feedback, inflating alpha or beta counts.

---

## 6. Performance Analysis

### PA-1: Correlation Tracker `update_correlations()` is O(n²) Unbounded
**File:** `correlation_tracker.py` lines 186-230

```python
def update_correlations(self):
    signals = list(self.history.keys())
    for i, sig_a in enumerate(signals):
        for sig_b in signals[i+1:]:  # O(n²) pairs
            corr = self.compute_correlation(sig_a, sig_b)
```

`compute_correlation()` (line 129) itself is O(m²) where m is the window size: it aligns observations with a double-nested loop scanning all of signal_b for each observation in signal_a (lines 152-164). Total complexity: O(n² × m²) per call to `update_correlations()`.

With 19 sources × 30 watchlist tickers = potentially hundreds of signal IDs, and window_size=30, this scales badly. The method is called from `detect_correlation_anomalies()` and `get_correlation_matrix()` — both called from the step loop. If signal count reaches 50+, this becomes a multi-second blocking operation in the main thread (no background execution).

### PA-2: Convergence Alerter `_prune_old_signals()` Called on Every `record_signal()`
**File:** `convergence_alerter.py` lines 253-263

```python
def _prune_old_signals(self):
    cutoff = datetime.now() - self.convergence_window
    for domain in self.signals:
        self.signals[domain] = [
            s for s in self.signals[domain]
            if s.timestamp >= cutoff
        ]
```

This is called from `record_signal()` (line 253), which is called for every signal in every fetch cycle. With 19 sources × 11 tickers and domain counts up to 16, this iterates all domains and rebuilds domain lists on every signal insert. This is O(total_signals_in_window) per insert rather than O(1).

For a 72-hour window running continuously, with signals arriving every few steps, the total signal count grows until the 72-hour point is reached. At peak, this creates measurable main-thread latency per signal.

### PA-3: StepTimer Sorts All Samples on Every `get_statistics()` Call
**File:** `step_timer.py` lines 53-62

```python
arr = sorted(times)  # O(n log n) sort of up to 1000 samples per call
```

`get_statistics()` is called for every StepTimer delegation in the HolonProxy sense cycle. With 1000 max_samples and multiple operations tracked, this performs multiple O(n log n) sorts per step. Not critical but adds ~1ms per step at full sample capacity.

### PA-4: Hypothesis Registry `find_by_trigger()` is O(n) Linear Scan
**File:** `hypothesis_registry.py` lines 150-159

```python
def find_by_trigger(self, source_a, source_b, lag_days) -> Optional[Hypothesis]:
    for hyp in self._hypotheses.values():  # O(n) full scan
        if hyp.status == HypothesisStatus.RETIRED:
            continue
        if (hyp.trigger.source_a == source_a ...):
            return hyp
```

Called by `hypothesis_generator.py` line 262 for every lag finding in `generate()`. With 43 lag findings and growing hypothesis count, this is O(43 × n_hypotheses) per generation cycle. Currently bounded (30-100 hypotheses), but grows over time without compaction.

### PA-5: Current Step Rate Observation
The `convergence_state.json` shows step 1600 at 11:24:34. The run_service.bat configures `--steps 2000`, suggesting steps complete in roughly the duration of the run. The convergence state shows `"hypotheses": {"active": 3, "probation": 0, "generated": 30, "promoted": 0}` — 30 generated hypotheses with 0 promotions in the current session suggests the hypothesis validator's gate is effectively blocking all promotions, not the generator. This could indicate gate over-tightening or DSR threshold issues rather than a performance bottleneck.

---

## 7. Operational Readiness

### OR-1: No Log Rotation
**File:** `run_service.bat`, no log configuration in `main.py`

`logging.basicConfig()` in `main.py` logs to stderr/stdout only. The bat file does not redirect logs to any file. Over continuous operation:
- `marathon-report.md` uses `open(out, "a")` — grows forever
- `run-log.md` uses `open(p, "a")` — grows forever
- `config_history.jsonl` — grows forever
- `discovery_log.jsonl` — grows forever
- `hypotheses.jsonl` — grows forever (compaction issue noted in PV-2)
- `thompson_history.jsonl` — grows forever

There is no size-based rotation, no daily rotation, no maximum file size enforcement anywhere. A system running continuously for months will accumulate GB of JSONL data.

### OR-2: No Stale Data Detection
The system has no mechanism to detect when market data is stale. The convergence_state.json timestamp ("2026-03-01T11:24:34") is overwritten every 100 steps, but nothing alerts if that file hasn't been updated in hours — which would indicate the step loop has stalled or the process has crashed.

The service bat file's 30-second restart handles crashes, but a hung thread (e.g., backtest running forever) would keep the process alive with a stalled step counter.

### OR-3: No Disk Space Guard
Multiple JSONL writers will continue writing until the disk is full. Python's file write will raise `OSError: [Errno 28] No space left on device`. This exception is caught at the innermost level (most writes have `except Exception: logger.debug(...)`) and silently swallowed. The system continues running but all persistence writes silently fail. On the next restart, the system finds empty or partially-written files and boots with factory state.

### OR-4: Partial State Recovery Is Undefined
There is no documented or tested recovery procedure for partial state (e.g., config_snapshot.json present but retirement_window.json missing, or pair_outcomes.json corrupted). The warm-start code in `market_systems.py` (line 24-29) only loads config snapshot. The retirement window loads in `HypothesisEngine.__init__()`, pair outcomes in `HypothesisGenerator.__init__()`. If any of these fail, the system boots with empty state and silently proceeds — potentially tightening meta-learning thresholds because the retirement rate is 0 (no data).

### OR-5: BacktestScheduler Uses Non-Atomic JSON Write for Critical Results
**File:** `backtest_scheduler.py` line 252

```python
with open(self._results_path, "w") as f:
    json.dump(results, f, indent=2)
```

This is a truncate-then-write, not an atomic rename. A crash mid-write leaves `sweep_backtest_results.json` in a partial state. On restart, `_is_stale()` calls `json.load(f)` which raises `JSONDecodeError`, and the method returns `True` (stale) — triggering an immediate backtest rerun on startup. This is the correct fallback behavior, but it means a crash during backtest write always triggers a 3-5 minute blocking operation immediately after restart, delaying the system coming back online.

---

## 8. Risk Matrix

| ID | Finding | Likelihood | Impact | Risk Level |
|----|---------|------------|--------|------------|
| CF-1 | Thompson history JSONL write without lock | High (already occurred once per MEMORY.md) | Critical — requires full Thompson rebuild | **CRITICAL** |
| CF-2 | Non-atomic write_text() for config/retirement/pair data | Medium (any crash triggers) | High — silent loss of meta-learned state | **HIGH** |
| CF-3 | Test data contaminating production pair_outcomes.json | Confirmed present | Medium — corrupts hypothesis priority ordering | **HIGH** |
| CF-4 | Retirement window seeded data misleads meta-learner | Medium | Medium — incorrect gate adjustment direction | **MEDIUM** |
| RC-1 | Discovery log write without lock from potential concurrent threads | Low (currently safe, fragile) | Medium — corrupted discovery log | **LOW** |
| RC-2 | Hypothesis engine background validation races with agent validation | Medium (agent requests frequent) | Medium — double-promote, double Thompson seed | **MEDIUM** |
| RC-3 | Thompson distributions dict mutated without lock | Medium (GIL mostly protects, not guaranteed) | Medium — stale distribution values | **MEDIUM** |
| RC-4 | JSONL appends non-atomic, partial lines on crash | Medium (any crash triggers) | Low — events skipped on replay, audit trail incomplete | **MEDIUM** |
| PV-1 | Config snapshot forward-compat gap silently drops new keys | Low (only affects schema changes) | Medium — new config keys not warm-started | **LOW** |
| PV-2 | Hypothesis registry grows unbounded, O(n) startup replay | Certain (continuous operation) | Medium — startup latency grows week-over-week | **MEDIUM** |
| PV-3 | Registered signals set grows unbounded | Certain (continuous operation) | Medium — registration batches slow over time | **MEDIUM** |
| PV-4 | Convergence alerter signals grow unbounded within window | Certain (continuous operation) | Low — memory grows, no cap within 72h | **LOW** |
| EG-1 | No backoff/circuit breaker for yfinance rate limiting | High (rate limits are real per MEMORY.md) | High — silent data gaps, Thompson learns nothing | **HIGH** |
| EG-2 | Backtest scheduler has no timeout on 3-5 min operation | Medium (network hangs possible) | Medium — silently stalled background thread | **MEDIUM** |
| EG-3 | Exception swallowing in round loop — persistent errors loop forever | Medium (state corruption from other bugs) | High — service churns on corrupted state | **HIGH** |
| EG-4 | run_service.bat loops on Python env failures | Low (env usually stable) | Medium — infinite 30s crash loop, no max retries | **MEDIUM** |
| EG-5 | OutcomeCollector registered set lost on crash — duplicate outcomes | Medium (any crash triggers) | Medium — Thompson double-counts some outcomes | **MEDIUM** |
| PA-1 | CorrelationTracker update_correlations() is O(n² × m²) | High (grows with signal diversity) | Medium — step latency grows with signal count | **MEDIUM** |
| PA-2 | Convergence alerter prunes on every record_signal() | Certain (by design) | Low — measurable but bounded latency | **LOW** |
| PA-3 | StepTimer sorts all samples every call | Certain (by design) | Low — ~1ms per step at full capacity | **LOW** |
| OR-1 | No log rotation | Certain (continuous operation) | Medium — disk fills over months | **MEDIUM** |
| OR-2 | No stale data detection | Certain (missing by design) | Medium — hung threads undetected | **MEDIUM** |
| OR-3 | No disk space guard | Certain (disk fills on long runs) | High — silent write failures, state loss | **HIGH** |
| OR-5 | Backtest result write non-atomic | Medium (any crash triggers) | Low — triggers immediate rerun, recovers | **LOW** |

---

## Appendix: Key File Evidence

**`data/market/pair_outcomes.json`** — Contains 8 test-fixture entries ("a|b", "a0|b0"..."a6|b6") alongside 2 real entries. Test isolation between test runs and production data is not enforced.

**`data/market/retirement_window.json`** — 50-entry window ending with 15 consecutive "retired" entries. `meta_promoted_total: 54`, `meta_retired_after_active: 2`. The 2 post-promotion retirements (3.7% false-positive rate) means Case A gate tightening is NOT triggered, but the retirement window shape (15 recent retirements) will eventually trigger Wire 2 threshold tightening when it crosses the 30-entry threshold for the retirement rate calculation.

**`data/market/config_snapshot.json`** — Shows `version: 3`, `modified_by: "meta_learner_retirement"` at `2026-02-28T19:36:44`. Meta-learning is functional. The snapshot's `hypothesis_gates` section is not visible in the truncated read but the version increment confirms the meta-learner has fired at least twice.

**`data/midge/convergence_state.json`** — Step 1600, "sideways" regime, all alerters null, 0 ticker alerts. The hypothesis count (3 active, 0 probation, 30 generated, 0 promoted) in the current session suggests either the hypothesis pipeline has not reached its generation cadence (500 steps) in a fresh session, or promotions are being blocked.

**`data/market/thompson_distributions.json`** — All entries show `"beta": 1.0` as the minimum, suggesting forgetting has brought many betas to floor. The `sec_edgar` entry shows `alpha: 1.49, beta: 1.0` — a very thin distribution (0.49 effective samples above prior) for what should be a well-observed source after 9,470+ total samples. This indicates the Bayesian forgetting is too aggressive for legacy key names (the real data is in `sec_form4`, not `sec_edgar`). The legacy keys in LEARNING_CONFIG exist for "backward compatibility" but their Thompson distributions are vestigial and effectively uninformed.
