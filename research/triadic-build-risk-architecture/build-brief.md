# Build Brief: Risk Architecture — DrawdownMonitor + SystemHealthMonitor + SelfMonitor

## Date: 2026-03-09
## Project: MIDGE

### Goal

Three monitors that protect MIDGE from catastrophic failure modes: capital loss (DrawdownMonitor), infrastructure degradation (SystemHealthMonitor), and behavioral runaway (SelfMonitor). These are the gate to live trading — without risk architecture, capital deployment is reckless.

### Build Tasks

**Round 1 (parallel — no file overlap):**

1. **DrawdownMonitor** — Track equity curve from realized P&L, calculate drawdown from peak, circuit-breaker halt when drawdown exceeds threshold. Persist equity history. Publish halt events on EventBus.

2. **SystemHealthMonitor** — Aggregate StepTimer latency data + per-subsystem error rates. Health tiers (Green/Yellow/Orange/Red). Auto-degrade subsystems with sustained failures. Publish health tier changes.

3. **SelfMonitor** — Track alert emission rate per direction. Detect behavioral anomalies: runaway alerting (>10 alerts per 100 steps), single-direction bias (>80% same direction over 50 alerts), confidence clustering (all alerts within 0.01 of each other = feedback loop). Circuit-break alert emission.

**Round 2 (after Round 1 complete):**

4. **Bootstrap Wiring** — Add channel constants, instantiate monitors in market_systems.py, wire into market_hooks.py step hooks, register triadic connections.

### Team Size: 3 builders (Round 1) + 1 builder (Round 2) + 2 reviewers

### Builder Assignments

| Builder | Domain | Files Owned |
|---------|--------|-------------|
| Drawdown Builder | Portfolio risk | `mae_core/market/intelligence/drawdown_monitor.py` (new), `tests/test_drawdown_monitor.py` (new) |
| Health Builder | System health | `mae_core/market/system_health_monitor.py` (new), `tests/test_system_health_monitor.py` (new) |
| Self Builder | Behavioral risk | `mae_core/market/intelligence/self_monitor.py` (new), `tests/test_self_monitor.py` (new) |
| Wiring Builder (Round 2) | Bootstrap | `mae_core/market/channels.py`, `mae_core/bootstrap/market_systems.py`, `mae_core/bootstrap/market_hooks.py`, `mae_core/bootstrap/market_connections.py` |

### Project Constraints
1. Never block the step loop — try/except around all monitor methods
2. No unbounded growth — equity history capped (rolling 10,000 entries), error counts use deque with maxlen
3. Zero regressions — `python -m pytest tests/ -v` must pass
4. Follow ResourceGovernor pattern — threading.RLock, EventBus injection, `get_statistics()` for HolonProxy
5. Tests required for all monitors
6. All monitors must work when dependencies are None (graceful degradation)
7. Law 1: No bare dyads — all monitors need triadic connections in market_connections.py

### Verification Plan
1. `python -m pytest tests/test_drawdown_monitor.py tests/test_system_health_monitor.py tests/test_self_monitor.py -v` — new tests pass
2. `python -m pytest tests/ -q --tb=line` — zero regressions
3. `python main.py --agents 3 --steps 30` — smoke test

---

## Technical Context for Builders

### Feature 1 — DrawdownMonitor (Drawdown Builder)

**What exists:**
- `PortfolioTracker` (portfolio_tracker.py) reads paper_trades.jsonl, marks positions to market, generates `ExitSignal` for stop-loss (-5%), take-profit (+15%), time-decay (30 days). `ClosedTrade` dataclass exists but `_closed` list is never populated.
- `KellyPositionSizer` (kelly_position_sizer.py) applies Kelly criterion. Half-Kelly (0.5x), 5% hard cap.
- Starting account value: 50,000 (from learning_config.py `PAPER_ACCOUNT_VALUE`).
- `ResourceGovernor` (resource_governor.py) is the gold-standard template for monitors.
- `CH_EXIT_SIGNAL` already defined in channels.py.
- midge-queue.md: "Define stop-loss threshold / circuit breaker (60% floor suggested)" — meaning halt if account drops below 60% of peak = -40% drawdown max.

**What to build:**
- `DrawdownMonitor` class with:
  - `__init__(self, event_bus=None, starting_capital=50_000.0, max_drawdown_pct=0.40, data_dir="data/market")` — follow ResourceGovernor pattern
  - `_lock: threading.RLock` for thread safety
  - `_peak_equity: float` — high water mark
  - `_current_equity: float` — starting_capital + sum of realized P&L
  - `_realized_pnl: float` — running total
  - `_equity_history: deque(maxlen=10_000)` — timestamped equity snapshots
  - `_trading_halted: bool` — circuit breaker flag
  - `record_trade_result(self, ticker: str, realized_pnl: float, direction: str = "")` — called when a position closes. Updates `_current_equity`, `_realized_pnl`, `_peak_equity`. Appends to `_equity_history`. Checks circuit breaker.
  - `get_current_drawdown(self) -> float` — returns (peak - current) / peak as percentage (0.0 = no drawdown, 0.4 = -40%)
  - `is_trading_halted(self) -> bool` — True if drawdown exceeds max_drawdown_pct
  - `get_statistics(self) -> dict` — for HolonProxy: peak_equity, current_equity, drawdown_pct, realized_pnl, trading_halted, trade_count
  - `save_state(self, path: str | None = None)` — atomic write equity history to JSONL
  - `load_state(self, path: str | None = None)` — restore from JSONL
- When drawdown exceeds 80% of max_drawdown_pct (i.e., 32% drawdown when max is 40%), publish `CH_DRAWDOWN_WARNING` on EventBus
- When drawdown exceeds max_drawdown_pct, publish `CH_TRADING_HALTED` on EventBus, set `_trading_halted = True`
- When equity recovers above max_drawdown_pct threshold, publish `CH_TRADING_RESUMED`, set `_trading_halted = False`
- Thread-safe: all state mutations under `_lock`

**Tests:**
- Record positive/negative trades, verify equity tracks correctly
- Peak equity updates only on new highs
- Drawdown calculation correct (percentage from peak)
- Circuit breaker triggers at max_drawdown_pct
- Warning fires at 80% of threshold
- Trading resumes when drawdown recovers
- Persistence round-trip (save/load)
- Thread safety (concurrent record_trade_result calls)
- Graceful degradation when event_bus is None
- get_statistics returns expected keys

### Feature 2 — SystemHealthMonitor (Health Builder)

**What exists:**
- `StepTimer` (step_timer.py) — tracks p50/p95/max latency per operation. Already wired into convergence_check, thompson_stats, velocity_scan, lag_correlation, granger_causality, post_mortem, thompson_calibration, hypothesis_engine.
- `DaemonMonitor` — PID, uptime, step_rate in heartbeat.json.
- Step hook try/except blocks in market_hooks.py swallow errors to logger.debug. No error counting.

**What to build:**
- `SystemHealthMonitor` class with:
  - `__init__(self, event_bus=None, step_timer=None, error_window=100, latency_threshold_ms=5000.0)` — follow ResourceGovernor pattern
  - `_lock: threading.RLock`
  - `_error_counts: dict[str, deque]` — per-subsystem error timestamps (deque maxlen=error_window)
  - `_subsystem_health: dict[str, str]` — "healthy", "degraded", "failed"
  - `_overall_tier: str` — "green", "yellow", "orange", "red"
  - `record_error(self, subsystem: str, error: Exception | None = None)` — append timestamp to subsystem deque. Re-evaluate health tier.
  - `record_success(self, subsystem: str)` — optional: reset consecutive error tracking for subsystem
  - `evaluate_health(self) -> str` — returns overall tier. Logic:
    - Green: all subsystems healthy
    - Yellow: 1-2 subsystems degraded (>5 errors in window)
    - Orange: 3+ subsystems degraded or any subsystem failed (>20 errors in window)
    - Red: core subsystem failed (convergence_check, thompson, sensing)
  - `is_degraded(self, subsystem: str) -> bool` — True if subsystem is degraded or failed
  - `get_latency_report(self) -> dict` — pulls data from StepTimer if available: per-operation p50/p95/max
  - `get_statistics(self) -> dict` — overall_tier, per-subsystem health, error counts, latency summary
- Define core subsystems: `CORE_SUBSYSTEMS = {"convergence_check", "thompson", "sensing", "outcome_evaluation"}`
- Publish `CH_HEALTH_TIER_CHANGE` on EventBus when tier changes (include old_tier, new_tier, degraded_subsystems)
- Thread-safe: all state mutations under `_lock`

**Tests:**
- Record errors, verify health tier transitions (green → yellow → orange → red)
- Core subsystem failure immediately goes to red
- Error window respects maxlen (old errors fall off)
- record_success resets subsystem health
- Latency report pulls from StepTimer
- get_statistics returns expected keys
- Graceful degradation when step_timer is None
- Thread safety

### Feature 3 — SelfMonitor (Self Builder)

**What exists:**
- Nothing. This is entirely new — no existing behavioral monitoring.
- AutoHealer pattern (auto_healer.py) has a `_self_monitor()` method that checks staleness, queue overflow, detection blindness — different concern but similar structure.

**What to build:**
- `SelfMonitor` class with:
  - `__init__(self, event_bus=None, rate_window=100, direction_window=50, max_alerts_per_window=10, bias_threshold=0.80)` — follow ResourceGovernor pattern
  - `_lock: threading.RLock`
  - `_recent_alerts: deque(maxlen=rate_window)` — timestamped alert records (direction, confidence, ticker, step)
  - `_alert_count: int` — total alerts observed
  - `_anomaly_flags: list[str]` — active anomaly reasons
  - `_alerting_suppressed: bool` — circuit breaker for alert emission
  - `record_alert(self, direction: str, confidence: float, ticker: str, step: int = 0)` — append to deque. Run anomaly checks.
  - `check_anomalies(self) -> list[str]` — returns list of anomaly strings:
    - `"runaway_rate"` — >max_alerts_per_window alerts in the deque
    - `"direction_bias"` — >bias_threshold of last direction_window alerts are same direction
    - `"confidence_clustering"` — std dev of last 20 confidences < 0.02 (suspiciously uniform = possible feedback loop)
    - `"ticker_flooding"` — single ticker accounts for >50% of last 30 alerts
  - `is_alerting_suppressed(self) -> bool` — True if any critical anomaly active (runaway_rate or confidence_clustering)
  - `reset_suppression(self)` — manual reset (for recovery after investigation)
  - `get_statistics(self) -> dict` — alert_count, active_anomalies, alerting_suppressed, recent_alert_rate, direction_distribution
- Publish `CH_BEHAVIORAL_ANOMALY` on EventBus when anomaly detected (include anomaly type, details)
- Auto-suppress alerting on `runaway_rate` or `confidence_clustering` (these indicate feedback loops)
- Do NOT auto-suppress on `direction_bias` or `ticker_flooding` (these may be legitimate market moves)
- Thread-safe: all state mutations under `_lock`

**Tests:**
- Normal alert flow, no anomalies detected
- Runaway rate detected when exceeding threshold
- Direction bias detected when >80% same direction
- Confidence clustering detected when std dev < 0.02
- Ticker flooding detected when single ticker dominates
- Auto-suppression triggers on runaway_rate
- Auto-suppression triggers on confidence_clustering
- No auto-suppression on direction_bias alone
- reset_suppression clears flag
- get_statistics returns expected keys
- Graceful degradation when event_bus is None
- Thread safety

### Round 2 — Bootstrap Wiring (Wiring Builder)

**Channel constants to add** (channels.py):
```python
CH_DRAWDOWN_WARNING = "market.risk.drawdown_warning"
CH_TRADING_HALTED = "market.risk.trading_halted"
CH_TRADING_RESUMED = "market.risk.trading_resumed"
CH_HEALTH_TIER_CHANGE = "market.health.tier_change"
CH_BEHAVIORAL_ANOMALY = "market.risk.behavioral_anomaly"
```

**market_systems.py additions:**
- Instantiate `DrawdownMonitor(event_bus=ctx.bus, data_dir="data/market")`
- Instantiate `SystemHealthMonitor(event_bus=ctx.bus, step_timer=ctx.step_timer)`
- Instantiate `SelfMonitor(event_bus=ctx.bus)`
- Load DrawdownMonitor state from `data/market/equity_history.jsonl`
- Store all three on ctx: `ctx.drawdown_monitor`, `ctx.system_health_monitor`, `ctx.self_monitor`

**market_hooks.py additions:**
- In `_market_sense_hook`: after paper trade gate, check `drawdown_monitor.is_trading_halted()` — skip paper trade if halted
- In `_market_sense_hook`: pass each convergence alert to `self_monitor.record_alert()`. Check `self_monitor.is_alerting_suppressed()` before emitting alerts.
- In try/except blocks: on exception, call `system_health_monitor.record_error(subsystem, exc)` instead of (or in addition to) `logger.debug`
- In `_daemon_persistence_flush`: save drawdown_monitor state

**market_connections.py additions:**
- Group 35: risk monitor triadic connections (drawdown↔bus↔portfolio, health↔bus↔timer, self↔bus↔alerter)

**Tests (in respective test files):**
- Wiring builder does NOT write new test files — the monitor builders handle their own tests
- But wiring builder should verify bootstrap doesn't break: `python main.py --agents 3 --steps 30`
