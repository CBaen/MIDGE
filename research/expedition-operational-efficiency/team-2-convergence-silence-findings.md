# Team 2: Convergence Silence — Research Findings

**Date:** 2026-03-12
**Researcher:** Expedition Researcher (Sonnet 4.6)
**Question:** Why does the convergence engine produce zero alerts despite 89K+ signals ingested?

---

## Critical Findings

### Finding 1: The Engine Has Fired — 5,494 Times

The most important fact: `data/market/alerter_state.json` shows `alert_counter: 5494`. The last alert times are:

```
bullish: 2026-03-12T05:07:34
bearish: 2026-03-12T05:07:34
```

At the time of this analysis (2026-03-12 ~21:40 ET), that is **16.6 hours ago**. The engine is not broken. It has been firing, and it fired today. The current silence is a **16-hour gap**, not a fundamental failure.

### Finding 2: The Signal Buffer Contains 117,706 Signals Across 13 Domains

`data/market/signal_buffer.json` (512KB):

| Domain | Signals | Bullish ≥0.6 | Bearish ≥0.6 |
|--------|---------|--------------|--------------|
| technical | 94,287 | 16,626 | 6,264 |
| events | 9,826 | 3,412 | 3,972 |
| insider | 6,738 | 832 | 46 |
| macro | 2,361 | 0 | 1 |
| institutional | 1,778 | 112 | 87 |
| energy | 1,102 | 258 | 584 |
| unknown | 751 | 0 | 543 |
| crypto | 305 | 0 | 0 |
| government | 167 | 39 | 125 |
| sentiment | 138 | 0 | 0 |
| positioning | 96 | 0 | 17 |
| news | 84 | 0 | 84 |
| contracts | 73 | 0 | 5 |

All 13 domains have live signals within their domain windows. Zero signals are expired.

### Finding 3: Bearish Convergence Should Be Firing Right Now

Simulating `_check_direction_convergence("bearish")` with the actual buffer:

**11 domains have bearish signals above min_strength=0.6:**
- unknown (eff=0.963), technical (0.963), energy (0.938), contracts (0.834), government (0.834), institutional (0.759), macro (0.746), insider (0.746), news (0.713), events (0.450), positioning (0.277)

With 11 qualifying domains vs min_domains=3, the dedup gap is 16.6h vs min_interval=4h — **a bearish alert should be generating on every step call**.

### Finding 4: The In-Memory Buffer Is NOT the Disk Buffer

This is the root cause of the reported "zero alerts during live daemon run" symptom.

The `signal_buffer.json` is the **persisted snapshot** written at daemon shutdown/flush. When the daemon restarts, `load_signal_buffer()` (in `convergence_buffer.py` line 126) must be called to restore it. But if the daemon ran for the session that produced these 117K signals and then stopped, the **next daemon restart starts with an empty in-memory buffer** until `load_signal_buffer()` is called AND until new signals arrive via `_collect_one()`.

The convergence check (`_market_sense_hook` in `market_hooks_steps_core.py` line 63) runs every step against `self.signals` — the **in-memory dict**, not the disk file. If the alerter was constructed fresh without loading the buffer, it starts empty.

### Finding 5: `load_signal_buffer()` Call — Present But Conditional

`convergence_buffer.py` provides `load_signal_buffer()`. It must be called during bootstrap. The question is whether the bootstrap calls it. The fact that `alert_counter=5494` and last alert is 05:07 today proves it ran earlier in this session. The 16.6h gap is a **step-rate/pace issue**, not a broken pipeline.

### Finding 6: DeepAnalyst Operates on Historical Archives, Not the Live Buffer

DeepAnalyst (`deep_analyst.py` line 114-117) uses `SignalArchiveReader` to load from `data/midge/signals/YYYY-MM-DD.jsonl` files (30-day lookback). It finds NOC/LMT/GD with 5-domain stacks because those tickers have signals across 30 days of archive files.

ConvergenceAlerter operates on `self.signals` — the **live rolling window** (72h for most domains). It is explicitly NOT per-ticker: `_check_direction_convergence` loops over all signals in the buffer regardless of ticker, picks the strongest per domain, and generates a global (not per-ticker) alert.

This is the architectural divergence: DeepAnalyst = historical synthesis engine. ConvergenceAlerter = live pattern detector. They look at different data with different time horizons. **Neither is broken.**

### Finding 7: The "Unknown" Domain Leak — 751 Bearish Signals

751 signals are in the `unknown` domain, all `bearish` with avg_strength=0.824. These have no entry in `domain_categories` dict (line 224-246, `convergence_alerter.py`). They count as a domain for convergence but contribute to a category of "unknown", which is not in the categories dict. This inflates domain count artificially and may corrupt category-based diversity calculations.

The source of "unknown" signals: `PatternLibrary._SOURCE_DOMAIN_MAP` (used by `sensing_collector.py` line 37) returns "unknown" for any source not in the map. These appear to be `finviz_insider_trades` or similar recently-added sources not yet mapped in `_SOURCE_DOMAIN_MAP`.

### Finding 8: Macro Domain — 2,361 Signals But Only 1 Bearish Above 0.6

Macro has 2,361 signals but 1,310 are neutral and only 1 bearish signal exceeds 0.6 strength. This means FRED macro signals are either firing as neutral or below the 0.6 strength threshold. For bullish: zero signals above 0.6. Macro is effectively absent from convergence despite high volume.

---

## Root Causes

### Root Cause 1: The Gap Is a Pace/Cadence Issue, Not a Dead Engine

The 16.6h silence since 05:07 is the primary symptom. With `--pace 1.0` daemon mode and convergence running every step, 16h of no alerts when 11 bearish domains qualify means either:
(a) The daemon is not currently running, OR
(b) The in-memory buffer does not match the disk buffer (buffer populated but alerter instance is different), OR
(c) A modifier is suppressing confidence below the alert threshold (not visible — no minimum confidence threshold exists in `_check_direction_convergence` — alerts fire regardless of confidence level)

Checking: there is **no minimum confidence threshold** on alert emission in `_check_direction_convergence`. The method returns an alert as long as `len(domains_seen) >= min_domains`. An alert with confidence=0.10 still fires. So dedup (4h window) is the only suppression gate.

With 16.6h since last alert and dedup requiring only 4h — the engine SHOULD be firing every 4 hours if the daemon is running.

### Root Cause 2: The Convergence Engine Is Global, Not Per-Ticker

ConvergenceAlerter treats all signals as one pool. When 11 domains fire bearish, it fires ONE bearish convergence alert regardless of which tickers the signals came from. If insider bought NOC, energy signal is about crude, and macro signal is about yields — those three fire as a global "bearish" alert with NOC as `primary_ticker` (whichever signal happens to have a symbol in metadata first).

This is why DeepAnalyst shows NOC/LMT/GD with 5-domain stacks but the convergence engine doesn't fire per-ticker alerts for them: the convergence engine doesn't do per-ticker convergence in `check_convergence()`. It does it in `convergence_ticker.py` (via `check_ticker_convergence()`), but that path is separate and must be explicitly called.

### Root Cause 3: Macro Domain Effectively Dead for Directional Signals

FRED client produces 2,361 signals but almost none are directional above 0.6. Yield curve and macro signals are being classified as "neutral" or at low strength. This eliminates one of the most reliable leading indicators from convergence stacks.

### Root Cause 4: Unknown Domain Contamination

751 signals in "unknown" domain inflate apparent domain count and escape the domain_categories mapping. This could cause a convergence alert to claim N+1 domains when one is actually "unknown." The `_SOURCE_DOMAIN_MAP` in `PatternLibrary` is missing entries for recently-wired sources.

---

## Recommended Fixes

### Fix 1: Confirm Daemon Status — Immediate

Run `python C:/Users/baenb/projects/MIDGE/main.py --daemon --pace 1.0` and check `daemon_output.log`. If the daemon has stopped, the silence is trivially explained. The alerter_state showing 16.6h gap is consistent with daemon shutdown at 05:07.

### Fix 2: Add Per-Ticker Convergence Surfacing to DeepAnalyst Output

DeepAnalyst's `analyze()` already groups by (ticker, direction) and finds 3+ domain stacks. The fix is to wire DeepAnalyst findings back into convergence alerter as ticker-specific synthetic signals, or to run `check_ticker_convergence()` from the signal buffer per-ticker. The architectural gap: per-ticker convergence exists in `convergence_ticker.py` but is not wired into the main sensing loop.

### Fix 3: Fix Macro Signal Directionality

`mae_core/market/apis/fred_client.py` and `fred_models.py` (currently staged/modified per git status) likely need to emit directional signals when yield curves invert, spreads widen/narrow, or fed funds rate changes direction. Currently FRED signals default to "neutral" for most series. This kills macro's contribution to convergence.

### Fix 4: Map Unknown Sources in `_SOURCE_DOMAIN_MAP`

Identify which sources produce "unknown" domain signals. Add them to `PatternLibrary._SOURCE_DOMAIN_MAP`. The 751 bearish "unknown" signals at 0.824 avg strength represent real signal power being mis-categorized.

### Fix 5: Verify `load_signal_buffer()` Is Called at Bootstrap

Confirm the market bootstrap calls `convergence_alerter.load_signal_buffer()` after construction. If it doesn't, every daemon restart loses the 72h signal history and must rebuild from new fetches — which takes hours and explains periodic silence after restarts.

---

## Gaps and Unknowns

1. **Is the daemon currently running?** The 16.6h gap is the biggest open question. Check `daemon_output.log` for last activity timestamp.

2. **What does `convergence_ticker.py` contain?** This file handles per-ticker convergence. It was not read in this investigation. If it has the per-ticker path that DeepAnalyst implies, the question is why it's not firing for NOC/LMT/GD.

3. **What confidence modifiers are active?** `_apply_confidence_modifiers` has 7 modifier paths (catalyst calendar, cross-asset, deception, economic calendar, archetype, pattern memory, combo Thompson). If economic calendar suppression is active (FOMC/CPI window), confidence is halved. This doesn't block alerts but would reduce reported confidence.

4. **Does `sensing_collector.py` call `check_convergence()` inline after `_collect_one()`?** The grep at line 204-211 shows it processes signals through PatternWatcher inline, but it was cut off before revealing whether it calls `check_convergence()` reactively. The `market_hooks_steps_core.py` already calls it every step — double-calling would be harmless but wasteful.

5. **FRED client changes (staged in git):** `mae_core/market/apis/fred_client.py` and `fred_models.py` are modified (staged). These changes may be fixing or breaking the FRED directionality issue. Their content determines whether Fix 3 is already in progress.

---

## Synthesis

**The convergence engine is not silent — it fired 5,494 times and was last active 16.6 hours ago.** The reported "zero alerts" almost certainly means either (a) the daemon is not currently running, or (b) the observation window started after the last alert at 05:07.

The architecture is sound: signals flow `_collect_one()` → `record_signal()` → `self.signals[domain]` → `check_convergence()` every step → `ctx._cached_alerts[0]`. With 117K signals in the buffer across 11 bearish domains above min_strength, the engine has everything it needs to fire on every 4-hour dedup cycle.

**The real gap DeepAnalyst reveals is architectural:** DeepAnalyst does per-ticker multi-domain synthesis. ConvergenceAlerter does global multi-domain synthesis. DeepAnalyst finding NOC/LMT/GD with 5-domain stacks is not surfaced as convergence alerts because ConvergenceAlerter picks one `primary_ticker` (whichever signal happens to carry a symbol in metadata) from a global pool — it doesn't say "NOC specifically has 5 domains converging."

**The highest-value fix is wiring per-ticker convergence** so that when DeepAnalyst or the signal buffer shows ticker X has 3+ independent domains, a ticker-specific alert fires. This turns the existing multi-domain signal wealth into actionable per-ticker inevitability alerts — which is exactly what Guiding Light described as MIDGE's purpose.

**Secondary priority:** Fix FRED macro directionality so 2,361 macro signals actually contribute to convergence instead of washing out as neutral.
