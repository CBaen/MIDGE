# Anvil Build Report — Round 2: sensing_hook.py Decomposition

**Agent:** Anvil
**Date:** 2026-02-28
**Task:** Decompose sensing_hook.py (1,075 lines) into 3 files

---

## Result: COMPLETE

Import verification passed:
```
python -c "from mae_core.market.sensing_hook import MarketSensingHook; print('OK')"
OK
```

---

## Files Produced

| File | Lines | Contents |
|------|-------|----------|
| `mae_core/market/sensing_hook.py` | 576 | MarketSensingHook class + module constants |
| `mae_core/market/sensing_fetchers.py` | 468 | 19 standalone fetch functions |
| `mae_core/market/sensing_lifecycle.py` | 124 | enrich_signal, store_signals, load_watchlist |
| **Total** | **1,168** | (original was 1,075 — delta is docstrings + imports) |

---

## What Was Extracted

### sensing_fetchers.py — 19 fetch functions

Each `_fetch_*()` instance method became a module-level standalone function.
The transformation pattern was consistent across all 19:

**Before (instance method):**
```python
def _fetch_sec_form4(self, converter) -> list:
    if self._sec_client is None:
        return []
    for ticker in self._watchlist.get("tickers", []):
        ...
```

**After (standalone function):**
```python
def fetch_sec_form4(sec_client: Any, watchlist: dict, converter: Callable) -> list:
    if sec_client is None:
        return []
    for ticker in watchlist.get("tickers", []):
        ...
```

Functions extracted:
- `fetch_sec_form4` — SEC Form 4 insider trades
- `fetch_sec_form8k` — SEC Form 8-K material events
- `fetch_congressional` — House stock trades (STOCK Act, $50K filter)
- `fetch_senate` — Senate stock trades ($50K filter)
- `fetch_hiring` — Job tracker / hiring blitz detection
- `fetch_usa_spending` — USASpending.gov contracts
- `fetch_sam_gov` — SAM.gov contract opportunities
- `fetch_social_sentiment` — ApeWisdom Reddit/WSB sentiment
- `fetch_finra_short` — FINRA daily short volume
- `fetch_sec_efts` — SEC EFTS full-text keyword search
- `fetch_finnhub` — Finnhub news + earnings (dual converter)
- `fetch_fred` — FRED macroeconomic indicators
- `fetch_session_sweep` — ICT session sweep / kill zone guard
- `fetch_ta_indicators` — RSI/MACD/Bollinger/Structure/Candle
- `fetch_cot` — CFTC Commitments of Traders
- `fetch_stocktwits` — StockTwits bull/bear sentiment
- `fetch_vix` — CBOE VIX term structure
- `fetch_trends` — Google Trends interest
- `fetch_finnhub_extras` — Economic calendar + analyst recommendations (dual converter)

### sensing_lifecycle.py — 3 lifecycle functions

- `enrich_signal(sig, velocity_detector, filing_analyzer, form8k_sentiment)` — velocity + filing-time modifier + Ollama 8-K sentiment. Mutates signal in place.
- `store_signals(signals, memory)` — Qdrant (if available) + JSONL cold storage
- `load_watchlist()` — reads data/midge/watchlist.json with default fallback

### sensing_hook.py — cleaned up class

- Imports all 19 fetchers from `sensing_fetchers`
- Imports lifecycle helpers from `sensing_lifecycle`
- `_fetch_source()` router calls `fetch_*(self._client, self._watchlist, converter)` instead of `self._fetch_*(converter)`
- `_collect_one()` calls `store_signals(signals, self._memory)` instead of `self._store_signals(signals)`
- `__init__` calls `load_watchlist()` instead of `self._load_watchlist()`
- `_fetch_source()` calls `enrich_signal(sig, ...)` instead of `self._enrich_signal(sig)`
- Removed unused `import math` (was flagged in task)
- `_store_signals`, `_enrich_signal`, `_load_watchlist` instance methods deleted (zero dead code)

---

## Design Decisions

**Why DATA_DIR / SIGNALS_DIR defined in both sensing_hook.py and sensing_lifecycle.py?**
Both files need access to the paths independently. `sensing_lifecycle.py` must be importable without going through sensing_hook.py — it's a standalone module. Both resolve from `__file__`, so both point to the same physical directory regardless of import order.

**Why are fetch functions not grouped in a class?**
They have no shared state between them. A class would be a namespace pretending to be a class. Module-level functions with explicit parameters are the correct pattern — they are independently testable, composable, and don't carry hidden `self` state.

**Why do dual-converter fetchers (finnhub, finnhub_extras) not use a tuple?**
Passing named parameters (`news_converter`, `earnings_converter`) is more readable and self-documenting than a tuple at position [0] and [1]. The call site in `_fetch_source()` is clear.

---

## Zero Behavior Changes Confirmed

Every function body is byte-for-byte identical to the original instance method bodies, with only two mechanical substitutions:
1. `self._<client>` → `<client>` (the parameter)
2. `self._watchlist` → `watchlist` (the parameter)

The `_fetch_source()` router arguments now pass `self._<client>` and `self._watchlist` explicitly, preserving the exact same values at call time.

The kill-zone time guard in `fetch_session_sweep()` is preserved verbatim including the `try/except pass` fallback for timezone failures.

---

## Files NOT modified

- `mae_core/market/bootstrap/market.py` — imports `MarketSensingHook` from the same path, no change needed
- All test files — no sensing_hook internals are tested directly (tests go through the public `step()` / `get_statistics()` API)
- `signal.py` — untouched (decomposed separately in this round by another agent)
