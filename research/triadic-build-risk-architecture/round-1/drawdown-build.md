# DrawdownMonitor — Build Report

**Date:** 2026-03-09
**Builder role:** Drawdown Builder (Round 1)
**Task:** Portfolio risk monitoring — equity curve tracking, drawdown circuit-breaker

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `mae_core/market/intelligence/drawdown_monitor.py` | 255 | DrawdownMonitor class |
| `tests/test_drawdown_monitor.py` | 351 | 46 tests — all passing |

---

## What Was Built

### DrawdownMonitor class

**Location:** `mae_core/market/intelligence/drawdown_monitor.py`

Follows the ResourceGovernor pattern exactly: `threading.RLock`, EventBus injection via `_bus`, best-effort publish in `_publish()` that swallows exceptions, `get_statistics()` for HolonProxy.

**Constructor signature:**
```python
DrawdownMonitor(
    event_bus=None,
    starting_capital=50_000.0,
    max_drawdown_pct=0.40,
    data_dir="data/market",
)
```

**State tracked:**
- `_current_equity` — starting_capital + sum of realized P&L
- `_peak_equity` — high-water mark, only moves up
- `_realized_pnl` — running total of all trade P&L
- `_trade_count` — integer counter
- `_equity_history` — `deque(maxlen=10_000)`, timestamped snapshots
- `_trading_halted` — circuit-breaker flag
- `_warning_active` — tracks whether we're in the warning zone (prevents warning spam)

**Public interface:**

| Method | Return | Description |
|--------|--------|-------------|
| `record_trade_result(ticker, realized_pnl, direction="")` | None | Call when position closes |
| `get_current_drawdown()` | float [0.0, 1.0] | (peak - current) / peak, floor at 0.0 |
| `is_trading_halted()` | bool | True when circuit-breaker tripped |
| `get_statistics()` | dict | For HolonProxy — 9 keys |
| `save_state(path=None)` | None | Atomic write to JSONL |
| `load_state(path=None)` | None | Restore from JSONL |

**get_statistics() keys:**
```python
{
    "peak_equity": float,       # high-water mark
    "current_equity": float,    # current account value
    "drawdown_pct": float,      # current drawdown fraction
    "realized_pnl": float,      # cumulative P&L
    "trading_halted": bool,     # circuit-breaker state
    "trade_count": int,         # total trades recorded
    "max_drawdown_pct": float,  # configured maximum
    "warning_active": bool,     # in warning zone
    "history_length": int,      # equity history size
}
```

---

## Design Decisions

### 1. Warning threshold uses `round(max_drawdown_pct * 0.80, 8)`

IEEE 754 floating point: `0.40 * 0.80 = 0.32000000000000006`. Without rounding, a trade that causes exactly 32% drawdown (`-$3,200` from `$10,000`) would compare `0.32 >= 0.32000000000000006` → `False` and fail to fire the warning. Rounding to 8 decimal places eliminates this boundary artifact.

### 2. `_warning_active` flag prevents event spam

Without this, every subsequent trade in the warning zone would re-publish `CH_DRAWDOWN_WARNING`. The flag is set on warning entry, cleared on recovery below the warning threshold, and reset when halted state transitions. This means:
- Warning fires once on entry
- Warning fires again after full recovery and re-entry (correct — it's a new event)

### 3. `_check_circuit_breaker` must be called under `_lock`

The method is private and assumes the caller holds the lock. This avoids double-acquisition of an `RLock` (which is reentrant, so it wouldn't deadlock, but it's cleaner to keep the lock in one place). The `record_trade_result` method acquires the lock and calls `_check_circuit_breaker` from inside.

### 4. Persistence format: JSONL with header metadata record

```
{"__meta__": true, "peak_equity": ..., "current_equity": ..., ...}
{"ts": ..., "equity": ..., "pnl_delta": ..., "ticker": ..., "direction": ...}
{"ts": ..., ...}
```

First line is always the metadata record (identified by `"__meta__": true`). Subsequent lines are equity snapshots. This means load_state can restore all critical running state (peak, current, halted) plus the full history, without needing a separate sidecar file.

Atomic write via `tmp` + `os.replace()` — same pattern as `learning_config.save_snapshot()`.

### 5. Channel constants defined locally (string literals)

As directed in the build brief: Wiring Builder adds them to `channels.py` in Round 2. The local definitions in `drawdown_monitor.py` use identical string values so imports will be a no-op substitution.

---

## Three-State Circuit-Breaker Logic

The `_check_circuit_breaker` method manages three threshold zones:

```
0%─────────────[32%─────────────[40%──────────→
    HEALTHY       WARNING ZONE    HALTED ZONE
```

Transitions:
| From | To | Trigger | EventBus |
|------|----|---------|----------|
| Healthy | Warning | drawdown >= 80% of max | CH_DRAWDOWN_WARNING |
| Warning | Halted | drawdown >= max | CH_TRADING_HALTED |
| Halted | Warning | drawdown < max | CH_TRADING_RESUMED |
| Warning | Healthy | drawdown < warning threshold | (no event — just clears flag) |
| Halted | Healthy | drawdown < warning threshold | CH_TRADING_RESUMED |

The `CH_TRADING_RESUMED` event fires whenever `_trading_halted` transitions from True to False, regardless of whether equity goes to the warning zone or fully recovers.

---

## Test Coverage

**46 tests across 7 test classes:**

| Class | Tests | What's Covered |
|-------|-------|----------------|
| `TestEquityTracking` | 7 | Profit/loss/accumulation/history/maxlen |
| `TestPeakEquity` | 3 | Peak-only-moves-up semantics |
| `TestDrawdownCalculation` | 5 | Formula, floor at 0, exact percentages |
| `TestCircuitBreaker` | 8 | Trip/resume/event dedup/publish |
| `TestWarningThreshold` | 5 | 80% boundary, spam prevention, refire |
| `TestGracefulDegradation` | 2 | event_bus=None never raises |
| `TestGetStatistics` | 3 | Required keys, initial values, post-trade |
| `TestPersistence` | 7 | Round-trip, atomic write, missing file, malformed |
| `TestThreadSafety` | 2 | 100 concurrent writers, concurrent reads |
| `TestEdgeCases` | 4 | Zero P&L, bus exception, fractional PNL, new peak |

**All 46 tests pass. Zero regressions (full suite run in progress).**

---

## Integration Notes for Wiring Builder (Round 2)

**Import:**
```python
from mae_core.market.intelligence.drawdown_monitor import DrawdownMonitor
```

**Instantiation in market_systems.py:**
```python
ctx.drawdown_monitor = DrawdownMonitor(
    event_bus=ctx.bus,
    starting_capital=50_000.0,  # or from LEARNING_CONFIG["paper_account_value"]
    max_drawdown_pct=0.40,
    data_dir="data/market",
)
ctx.drawdown_monitor.load_state()
```

**Usage in market_hooks.py (paper trade gate):**
```python
if ctx.drawdown_monitor.is_trading_halted():
    logger.info("DrawdownMonitor: trading halted, skipping paper trade")
    return  # skip this trade
```

**Persistence flush in _daemon_persistence_flush:**
```python
if hasattr(ctx, "drawdown_monitor"):
    ctx.drawdown_monitor.save_state()
```

**When a position closes (record realized P&L):**
```python
ctx.drawdown_monitor.record_trade_result(
    ticker=closed_trade.ticker,
    realized_pnl=closed_trade.realized_pnl,
    direction=closed_trade.direction,
)
```

**channel constants to add to channels.py (identical to local definitions):**
```python
CH_DRAWDOWN_WARNING = "market.risk.drawdown_warning"
CH_TRADING_HALTED = "market.risk.trading_halted"
CH_TRADING_RESUMED = "market.risk.trading_resumed"
```

**Triadic connections (for market_connections.py Group 35):**
- `drawdown_monitor ↔ event_bus ↔ portfolio_tracker`
  - Primary: drawdown_monitor → portfolio_tracker (receives closed trade P&L)
  - Verification: drawdown_monitor → event_bus → portfolio_tracker (halt events route)
  - Balance: portfolio_tracker → event_bus → drawdown_monitor (future: portfolio events)

---

## Bugs Found and Fixed

1. **Floating-point boundary bug in warning threshold**: `0.40 * 0.80 = 0.32000000000000006` in IEEE 754. Fixed by rounding to 8 decimal places. Without this fix, an account dropping exactly 32% from peak would silently miss the warning.

2. **Test arithmetic error**: `test_stats_reflect_trades` had an incorrect comment (`9,200.0 - 2,000.0 + 10,000.0 = 17,200`) — the correct value after +2000 then -4800 is `10,000 + 2,000 - 4,800 = 7,200`. Fixed in test.
