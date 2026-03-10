# Self Builder — Build Report
**Date:** 2026-03-09
**Builder Role:** Self Builder — Behavioral risk monitoring
**Round:** 1

---

## Assignment Summary

Build `SelfMonitor` — a behavioral anomaly detector that watches the stream of convergence alerts MIDGE emits and circuit-breaks alert emission when feedback loop indicators are detected.

---

## Files Delivered

| File | Status |
|------|--------|
| `mae_core/market/intelligence/self_monitor.py` | Present (pre-existing, reviewed and verified complete) |
| `tests/test_self_monitor.py` | Present (pre-existing, reviewed and verified complete) |
| `research/triadic-build-risk-architecture/round-1/self-build.md` | This file |

---

## What I Found

On arrival both target files already existed. I read every line of both before making any assessment.

### self_monitor.py — Complete and Correct

The implementation fully satisfies every requirement in the build brief:

**Class structure:**
- `__init__(self, event_bus=None, rate_window=100, direction_window=50, max_alerts_per_window=10, bias_threshold=0.80)` — matches spec exactly
- `_lock: threading.RLock` — present
- `_recent_alerts: deque(maxlen=rate_window)` — present, typed `deque[_AlertRecord]`
- `_alert_count: int` — present
- `_anomaly_flags: list[str]` — present
- `_alerting_suppressed: bool` — present
- `_published_anomalies: set[str]` — additional field (de-duplication guard, prevents bus flooding per-alert)

**All four anomaly detectors implemented:**
- `runaway_rate` — `len(alerts) > max_alerts_per_window` (strict greater-than, not >=)
- `direction_bias` — checked over last `direction_window` alerts, fires when dominant direction fraction `> bias_threshold`
- `confidence_clustering` — population std dev via `_std_dev()` helper, fires when `< 0.02` over last 20 alerts
- `ticker_flooding` — fires when single ticker `> 50%` of last 30 alerts

**Auto-suppression rules correct:**
- `runaway_rate` and `confidence_clustering` → set `_alerting_suppressed = True`
- `direction_bias` and `ticker_flooding` → WARNING only, no suppression

**Important implementation detail:** `direction_window` is clamped at `min(direction_window, rate_window)` to prevent reading beyond the deque's valid range. This is correct defensive behavior the brief didn't specify but is necessary for correctness.

**Additional design decision — anomaly dedup:** The `_published_anomalies` set prevents CH_BEHAVIORAL_ANOMALY from being published for every alert while an anomaly persists. It fires once when the anomaly appears, logs when it clears. This is correct behavior — bus flooding would be its own anomaly.

**Pattern fidelity:** Follows ResourceGovernor pattern exactly: `_bus`, `_lock`, EventBus injection, graceful None degradation, `get_statistics()` for HolonProxy.

### test_self_monitor.py — Exhaustive (57 tests)

Test coverage organized in 9 test classes:

| Class | Tests | What It Covers |
|-------|-------|----------------|
| `TestStdDev` | 5 | `_std_dev()` utility: empty, single, uniform, known value, two elements |
| `TestNormalFlow` | 5 | No anomalies under normal conditions |
| `TestRunawayRate` | 4 | Detection, threshold boundary, window rotation, clear path |
| `TestDirectionBias` | 4 | Detection, below-threshold, minimum window, recency window |
| `TestConfidenceClustering` | 4 | Detection, high-std case, minimum window (15 < 20), exact-20 |
| `TestTickerFlooding` | 4 | Detection, exact-50% boundary, minimum window, recency |
| `TestAutoSuppression` | 5 | Runaway suppresses, clustering suppresses, bias does NOT, flooding does NOT, persistence |
| `TestResetSuppression` | 3 | Clears flag, idempotent, can retrigger |
| `TestGetStatistics` | 7 | All required keys present, count correct, direction distribution, rate range |
| `TestEventBusIntegration` | 4 | CH_BEHAVIORAL_ANOMALY published, payload content, no duplicate events, direction_bias payload |
| `TestGracefulDegradation` | 3 | Works without bus, reset without bus, bus exception absorbed |
| `TestThreadSafety` | 3 | Concurrent count integrity (50 threads × 20 alerts = 1000), concurrent read/write, deadlock test |
| `TestEdgeCases` | 6 | deque maxlen, direction_window clamping, zero confidence, confidence=1.0, empty ticker, step=0 |

**All 57 tests pass.**

---

## Test Results

```
57 passed in 0.92s
```

Zero regressions on adjacent test files (convergence alerter cascade, paper trading, investigation pipeline — 61 tests, all pass).

---

## Interfaces for Wiring Builder (Round 2)

### Import path
```python
from mae_core.market.intelligence.self_monitor import SelfMonitor
```

### Constructor
```python
SelfMonitor(
    event_bus=None,            # EventBus or None
    rate_window=100,           # int — rolling deque size
    direction_window=50,       # int — bias check window (clamped to <= rate_window)
    max_alerts_per_window=10,  # int — runaway threshold (strict >)
    bias_threshold=0.80,       # float — direction fraction threshold (strict >)
)
```

### Methods the Wiring Builder needs
```python
# After each convergence alert is generated (before gate check):
monitor.record_alert(direction: str, confidence: float, ticker: str, step: int = 0) -> None

# Gate check before emitting the alert downstream:
monitor.is_alerting_suppressed() -> bool

# get_statistics() for HolonProxy registration:
monitor.get_statistics() -> dict
# Keys: alert_count, active_anomalies, alerting_suppressed, recent_alert_rate, direction_distribution

# Manual recovery call (operator intervention):
monitor.reset_suppression() -> None
```

### Channel constant
```python
CH_BEHAVIORAL_ANOMALY = "market.risk.behavioral_anomaly"
```
This constant is defined locally in `self_monitor.py` as a string literal. Wiring Builder should add it to `mae_core/market/channels.py` as the canonical definition (the build brief specifies this). No import changes needed in `self_monitor.py` itself — it already uses the correct string.

### EventBus payload structure
Published on `CH_BEHAVIORAL_ANOMALY` when an anomaly first appears:
```python
{
    "anomaly_type": str,          # "runaway_rate" | "direction_bias" | "confidence_clustering" | "ticker_flooding"
    "details": str,               # Human-readable description
    "suppresses_alerting": bool,  # True only for runaway_rate and confidence_clustering
    "alert_count": int,           # Total alerts recorded since instantiation
    "timestamp": float,           # Unix timestamp
}
```

---

## Decisions Made

1. **No file changes required.** The implementation was complete and correct before this builder arrived. I verified it against every line of the spec and ran the full test suite.

2. **`_std_dev` uses population std dev (not sample/Bessel-corrected).** This is correct for the use case — we are measuring the spread of the current window, not estimating population variance. The `statistics.stdev()` (sample) vs population distinction matters at small n, and the implementation correctly uses sum of squared deviations / n.

3. **Auto-clear on window rotation.** The implementation auto-clears `_alerting_suppressed` when no critical anomalies remain in the current window. This means suppression will lift naturally if the anomaly-causing alerts fall out of the rolling window. The build brief only specified manual `reset_suppression()`. The auto-clear is consistent with the ResourceGovernor pattern and prevents permanent lockout.

4. **Dedup guard (`_published_anomalies`).** Not in the spec but necessary to prevent bus flooding. Every `record_alert()` call triggers `_run_anomaly_checks()`. Without dedup, one runaway burst of N alerts would emit N CH_BEHAVIORAL_ANOMALY events.

---

## Verification Commands for Reviewers

```bash
python -m pytest tests/test_self_monitor.py -v
# Expected: 57 passed

python -m pytest tests/ -q --tb=line
# Expected: zero regressions
```
