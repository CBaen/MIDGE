# Anvil Round 1 Research: Convergence + Signal Pipeline Fixes

**Role:** Anvil — Convergence + Signal Pipeline Builder
**Round:** 1 — Research and preparation only (no code changes)
**Date:** 2026-03-01
**Scope:** P4B (alert dedup race) and P3 (direction-agnostic finra_short) in `convergence_alerter.py`

---

## Research Task 1: finra_short Neutral-Signal Path (P3)

### What `from_short_interest()` Emits

File: `C:\Users\baenb\projects\MIDGE\mae_core\market\signal_adapters\market_data.py`
Lines 15-57.

The adapter emits `direction="neutral"` for **all** short ratio values below 0.6:

```python
if data.short_ratio >= 0.6:
    direction = "bearish"  # Heavy shorting pressure
elif data.short_ratio >= 0.4:
    direction = "neutral"  # Elevated but not extreme
else:
    direction = "neutral"
```

This means:
- `short_ratio >= 0.6` → `direction = "bearish"`
- `short_ratio < 0.6` → `direction = "neutral"` (this covers the majority of cases, including elevated short interest that is genuinely actionable)

The `outcome_symbol` is set to `data.symbol`. The `domain` is hardcoded to `"institutional"`. The `source` is `"finra_short"`.

### What the OutcomeCollector Does With "neutral"

File: `C:\Users\baenb\projects\MIDGE\mae_core\market\intelligence\outcome_collector.py`
Line 106.

```python
direction = {"bullish": "up", "bearish": "down"}.get(sig.direction, "")
```

A signal with `direction="neutral"` maps to `direction=""` (empty string) when passed to `OutcomeTracker.record_prediction()`. This is correct behavior — it registers the prediction as direction-agnostic, meaning success is judged by **magnitude alone** (5% move in either direction).

This is confirmed by `outcome_tracker.py` lines 294-300:

```python
if direction == "up":
    direction_ok = pct_change > 0
elif direction == "down":
    direction_ok = pct_change < 0
else:
    # No direction specified — magnitude alone determines success
    direction_ok = True
```

So the `OutcomeCollector` → `OutcomeTracker` pipeline correctly handles `direction="neutral"` by registering it as `direction=""` and evaluating purely on magnitude. This explains why `outcomes.jsonl` records for `finra_short` all show `"direction": ""`.

### Where "neutral" Signals Get Dropped From Convergence

File: `C:\Users\baenb\projects\MIDGE\mae_core\market\intelligence\convergence_alerter.py`
Lines 458-462 in `_check_direction_convergence()`.

```python
matching = [
    s for s in signals
    if s.direction == direction and s.strength >= self.min_strength
]
```

`_check_direction_convergence()` is called with `direction="bullish"` or `direction="bearish"`. The filter `s.direction == direction` means:
- A `finra_short` signal with `direction="neutral"` is **excluded from both bullish and bearish convergence checks**.
- It never contributes to domain convergence.
- It never gets its strength tallied.
- The `"institutional"` domain it occupies never appears in `domains_seen`.

The `get_domain_status()` method at lines 529-555 also buckles out neutral signals: it computes `bullish` and `bearish` sub-lists by direction, so a neutral-direction signal contributes to neither count. The dominant direction for the domain is computed from the sub-lists, and a domain with only neutral signals shows as `dominant = "neutral"` which then makes `matrix[domain_a][domain_b] = 0` (no agreement).

### Summary of the Drop

A finra_short signal with `short_ratio=0.5` follows this path:
1. `from_short_interest()` → `direction="neutral"`, `domain="institutional"`, stored in `self.signals["institutional"]`
2. `check_convergence()` calls `_check_direction_convergence("bullish")` and `_check_direction_convergence("bearish")`
3. In both calls, `s.direction == "bullish"` and `s.direction == "bearish"` are both `False` for this signal
4. The signal is excluded from `converging_signals`
5. `"institutional"` does not appear in `domains_seen`
6. The min_domains=3 check counts one fewer domain
7. The signal's Thompson weight and confidence are never factored in
8. The signal's strength contributes nothing to the convergence score

Only when `short_ratio >= 0.6` (where the adapter emits `direction="bearish"`) does finra_short participate in convergence — and then only in the bearish check. This is wrong: finra_short with high short ratio is a volatility predictor (squeeze potential OR continued short pressure), not purely a bearish directional signal.

### The Fix for P3

There are two complementary fixes needed:

**Fix A: Correct the signal adapter's direction logic**

In `from_short_interest()`, the directional logic is wrong. A high short ratio is not cleanly "bearish" — it signals high volatility risk (could squeeze bullish OR continue bearish). The adapter should emit `direction="neutral"` for all cases and rely on strength to convey magnitude. This preserves the correct OutcomeCollector→Thompson→magnitude-only evaluation path.

Change lines 22-24 in `market_data.py` from:
```python
if data.short_ratio >= 0.6:
    direction = "bearish"  # Heavy shorting pressure
elif data.short_ratio >= 0.4:
    direction = "neutral"  # Elevated but not extreme
else:
    direction = "neutral"
```
To:
```python
direction = "neutral"  # Direction-agnostic: predicts volatility, not direction
```

**Fix B: Handle direction-agnostic signals in `_check_direction_convergence()`**

When `direction="neutral"`, a signal should be counted as contributing context to convergence without polluting directional purity. The most surgical fix is to include neutral signals from known direction-agnostic sources as supporting evidence in both bullish and bearish checks, but with their contribution clearly labeled.

The minimal fix: in `_check_direction_convergence()`, after collecting directionally matching signals, also collect neutral signals from the same domain that meet the strength threshold, and include them in `converging_signals` at their face value. This lets the "institutional" domain slot count when finra_short is present:

```python
# After collecting directional matching signals:
neutral_matching = [
    s for s in signals
    if s.direction == "neutral" and s.strength >= self.min_strength
]
# Include neutral signals as supporting context (direction-agnostic sources)
if neutral_matching and not matching:
    # Domain had no directional signal but has a strong neutral signal
    # Include as contextual support — doesn't anchor direction but adds a domain
    strongest_neutral = max(neutral_matching, key=lambda s: s.strength)
    converging_signals.append(strongest_neutral)
    domains_seen.add(domain)
    category = self.domain_categories.get(domain, domain)
    categories_seen.add(category)
```

This approach correctly reflects the semantics: finra_short is saying "something is about to happen on this stock" without specifying direction. Its contribution to a bullish convergence is "the stock is primed for a move, and multiple other domains say bullish."

**Which sources are affected:**
- `finra_short` (domain: "institutional", currently always neutral except short_ratio >= 0.6)
- `fred_macro` (domain: "macro", also sometimes emits neutral)
- `vix_term_structure` (domain: "volatility", neutral when VIX is flat)
- Pre-earnings `finnhub_earnings` (domain: "events", neutral before earnings)

**Important constraint:** A neutral signal should only contribute to convergence if there is at least one directional signal from another domain first. Do not allow a collection of only neutral signals to produce a directional alert.

---

## Research Task 2: Alert Dedup Race (P4B)

### The Dedup Guard

File: `C:\Users\baenb\projects\MIDGE\mae_core\market\intelligence\convergence_alerter.py`
Lines 424-437 in `check_convergence()`.

```python
# Alert deduplication — suppress re-alert within interval
now = datetime.now()
filtered = []
for alert in alerts:
    direction = alert.direction if hasattr(alert, "direction") else "neutral"
    if (self._last_alert_direction == direction
            and self._last_alert_time is not None
            and (now - self._last_alert_time).total_seconds() / 3600
            < self._min_alert_interval_hours):
        continue  # Suppress — same condition, too recent
    filtered.append(alert)
    self._last_alert_direction = direction
    self._last_alert_time = now
alerts = filtered
```

### The Race Condition

**State variables** (lines 199-201):
```python
self._last_alert_direction = None
self._last_alert_time = None
self._min_alert_interval_hours = 4.0
```

These are plain instance attributes with no thread protection.

**The root cause is not threading — it is sequential logic within a single call.**

When `check_convergence()` is called and both bullish and bearish convergence are active, `alerts` contains two alerts. The loop processes them sequentially:

1. `alert[0]` (bullish): `_last_alert_direction` is None → condition fails → alert passes through → `_last_alert_direction = "bullish"`, `_last_alert_time = now`
2. `alert[1]` (bearish): `_last_alert_direction == "bullish"` which != "bearish" → condition fails → alert passes through → `_last_alert_direction = "bearish"`, `_last_alert_time = now`

Both alerts pass. But the real bug is that the dedup only tracks **one** direction at a time. After the first call where bullish fires, `_last_alert_direction = "bullish"`. On the next call, bearish fires: `"bullish" != "bearish"` → bearish passes. Bearish is now stored. On the next call, bullish fires again: `"bearish" != "bullish"` → bullish passes. The two directions leapfrog each other infinitely, both fire on every call, and the `_min_alert_interval_hours = 4.0` check never suppresses anything because the direction always differs.

**Secondary race:** If threading is ever added (or if `step()` calls `check_convergence()` which then gets called separately by the step hook), two simultaneous callers can both pass the time check before either updates `_last_alert_time`. This is a classic check-then-act race.

### Evidence from Production

`data/market/discovery_log.jsonl` contains CONV-20260227-0001 through CONV-20260227-0021 (20+ identical alerts logged in production in the same session). The `_alert_counter` increments on each call to `_check_direction_convergence()` (line 503), which is called inside `check_convergence()`. The step hook calls `check_convergence()` every step at line 216 of `market_hooks.py`. So if bullish convergence is active for 20+ consecutive steps, 20 alerts are produced.

### All Callsites of `check_convergence()`

From the codebase:

1. **`mae_core/bootstrap/market_hooks.py` line 216/218**: Called every single step inside `_market_sense_hook()`. This is the primary source of the storm.
2. **`mae_core/bootstrap/market_hooks.py` line 525**: Called every 10 steps for each of 3 tiered alerters (tactical/strategic/thematic) inside `_sensing_step_with_advisory()`.
3. **`mae_core/market/intelligence/convergence_alerter.py` line 732**: Called inside `step()` method, which is the HolonProxy delegation hook — unclear if this is also hooked into the step loop separately, but it would be a double-call if so.
4. **`mae_core/market/step_timer.py` line 28**: Used in a test/diagnostic context.
5. **`midge_scan.py` lines 387/397**: CLI tool, separate process.

The dangerous path is #1 (every step) + #2 (every 10 steps for tiered alerters). If conditions persist for N steps, `N` global alerts and `N/10 * 3` tiered alerts are generated.

### The Fix for P4B

**Fix 1: Per-direction last-alert tracking**

Replace the single `_last_alert_direction`/`_last_alert_time` pair with a dict keyed by direction:

```python
# In __init__():
self._last_alert_times: Dict[str, Optional[datetime]] = {}  # direction -> last alert time
```

Remove lines 199-201 (the old scalar vars).

Replace the dedup loop (lines 424-437) with:

```python
now = datetime.now()
filtered = []
for alert in alerts:
    direction = alert.direction if hasattr(alert, "direction") else "neutral"
    last_time = self._last_alert_times.get(direction)
    if (last_time is not None
            and (now - last_time).total_seconds() / 3600
            < self._min_alert_interval_hours):
        continue  # Suppress — same direction, too recent
    filtered.append(alert)
    self._last_alert_times[direction] = now
alerts = filtered
```

This correctly tracks bullish and bearish suppression independently. Both can be suppressed simultaneously. Neither suppresses the other.

**Fix 2: Add `threading.Lock()` for thread safety**

Even though the current primary race is not inter-thread, the dedup should be made thread-safe for the day when the step hook and tiered alerters could be called from different execution contexts. In `__init__()`:

```python
import threading
self._dedup_lock = threading.Lock()
```

Wrap the dedup block:

```python
with self._dedup_lock:
    now = datetime.now()
    filtered = []
    for alert in alerts:
        direction = alert.direction if hasattr(alert, "direction") else "neutral"
        last_time = self._last_alert_times.get(direction)
        if (last_time is not None
                and (now - last_time).total_seconds() / 3600
                < self._min_alert_interval_hours):
            continue
        filtered.append(alert)
        self._last_alert_times[direction] = now
    alerts = filtered
```

**Fix 3: Move alert counter inside `_check_direction_convergence()` to only increment on non-suppressed alerts**

Currently `_alert_counter` increments at line 503 inside `_check_direction_convergence()` before the dedup check runs. This means duplicate alerts still increment the counter and consume CONV-YYYYMMDD-NNNN IDs. Move the counter increment to after the dedup filter passes. This requires either passing a counter flag or returning the alert and incrementing in `check_convergence()` only when the alert is not suppressed.

The cleanest approach: move `self._alert_counter += 1` out of `_check_direction_convergence()` and into `check_convergence()` after the dedup passes. Change `_check_direction_convergence()` to not assign an `alert_id` and instead return the partially-constructed alert, then assign the ID in `check_convergence()` only when it passes dedup.

Alternatively (simpler): keep the counter increment where it is but change the `alert_id` format to use a UUID or hash, making duplicate suppression irrelevant to the ID namespace. The simplest fix is just fixing the per-direction dedup — the wasted counter IDs are cosmetic, not functional.

**The minimum viable fix:**
- Replace scalar `_last_alert_direction`/`_last_alert_time` with `_last_alert_times: Dict[str, Optional[datetime]]`
- Add `threading.Lock()` around the check-and-update block
- This eliminates the leapfrog bug and the thread race

**What this does NOT fix:** The `step()` method on `ConvergenceAlerter` (line 732) calls `check_convergence()` directly. If this `step()` is wired as a HolonProxy delegation target AND the bootstrap step hook also calls `check_convergence()`, there will be two calls per step. The dedup will suppress the second call's results, which is correct behavior — but the double-call is wasteful. Verify in bootstrap that `convergence_alerter.step()` is not separately hooked.

---

## Research Task 3: Convergence Window for P5B (Per-Domain Windows)

### How `convergence_window_hours` Works

File: `C:\Users\baenb\projects\MIDGE\mae_core\market\intelligence\convergence_alerter.py`

**Constructor** (line 145, 163):
```python
convergence_window_hours: int = 72,
...
self.convergence_window = timedelta(hours=convergence_window_hours)
```

A single `timedelta` is stored. All domains use the same window.

**`_prune_old_signals()`** (lines 255-263):
```python
def _prune_old_signals(self):
    """Remove signals outside the convergence window."""
    cutoff = datetime.now() - self.convergence_window

    for domain in self.signals:
        self.signals[domain] = [
            s for s in self.signals[domain]
            if s.timestamp >= cutoff
        ]
```

All domains are pruned at the same cutoff. A COT signal with `timestamp = 5 days ago` and a `convergence_window = 72 hours` is pruned — even though COT data is weekly and a 5-day-old positioning signal is still valid.

`_prune_old_signals()` is called from:
- `record_signal()` (line 253) — on every new signal ingestion
- `check_convergence()` (line 408) — at the start of every convergence check
- `get_domain_status()` (line 526) — on status queries
- `check_ticker_convergence()` (line 572) — on per-ticker convergence

### All Domain Values Signals Can Have

From the `domain_categories` dict (lines 176-194) and from signal adapter review:

**Behavioral:** `insider`, `congress`
**Market:** `crypto`, `technical`, `volume`, `volatility`
**Social:** `sentiment`, `reddit`
**Information:** `news`, `events`
**Financial:** `fundamentals`, `macro`
**Institutional:** `government`, `contracts`, `institutional_synthesis`, `positioning`, `institutional`

Note: `"institutional"` is used by `finra_short` (see `market_data.py` line 42) but does NOT appear in `domain_categories`. This means `self.domain_categories.get("institutional", "institutional")` returns `"institutional"` as the category (falls back to domain name). This is a latent issue — the `finra_short` signals' domain is uncategorized.

### What Needs to Change for P5B

**Design needed:**

Replace the single `self.convergence_window` with a dict of per-domain windows:

```python
# New parameter in __init__():
domain_windows: Optional[Dict[str, int]] = None  # domain -> hours override

# In __init__():
self._domain_windows: Dict[str, timedelta] = {}
if domain_windows:
    for domain, hours in domain_windows.items():
        self._domain_windows[domain] = timedelta(hours=hours)
self._default_window = timedelta(hours=convergence_window_hours)
```

Update `_prune_old_signals()`:

```python
def _prune_old_signals(self):
    """Remove signals outside the convergence window (per-domain)."""
    now = datetime.now()
    for domain in self.signals:
        window = self._domain_windows.get(domain, self._default_window)
        cutoff = now - window
        self.signals[domain] = [
            s for s in self.signals[domain]
            if s.timestamp >= cutoff
        ]
```

**Recommended domain windows** (from deliverable.md P5B):
- `positioning`: 14 days (336 hours) — COT is weekly, positioning thesis lasts weeks
- `government`: 7 days (168 hours) — Congressional trades have 30-45 day disclosure lag, the signal itself is strategic
- `contracts`: 7 days (168 hours) — Contract awards are thesis-level events
- All others: 72 hours (default)

**Bootstrap wiring:**
In `_wire_sensing_hook()` at `market_hooks.py`, the `ConvergenceAlerter` instantiation for the global alerter passes no `domain_windows`. In Round 2 this would be added:

```python
# In bootstrap/market.py or bootstrap/market_systems.py where ConvergenceAlerter is built:
domain_windows={
    "positioning": 14 * 24,   # 336 hours
    "government": 7 * 24,     # 168 hours
    "contracts": 7 * 24,      # 168 hours
}
```

The tiered alerters in `market_hooks.py` lines 405-414 would NOT receive per-domain windows — their entire window is already narrowed (tactical=48h, strategic=21d, thematic=90d) which subsumes the per-domain logic.

**Note:** `self.convergence_window` is referenced in `to_dict()` at line 740. This needs updating to expose the per-domain dict:

```python
"convergence_window_hours": {
    "default": self._default_window.total_seconds() / 3600,
    **{d: w.total_seconds() / 3600 for d, w in self._domain_windows.items()}
}
```

---

## Cross-Cutting Notes for Round 2

### Missing `"institutional"` from `domain_categories`

`finra_short` signals use `domain="institutional"` (line 42, `market_data.py`). The `domain_categories` dict (lines 176-194, `convergence_alerter.py`) does not contain `"institutional"`. This means `self.domain_categories.get("institutional", "institutional")` returns the domain name itself as the category — which means it is effectively its own category for cross-domain counting purposes. This is not harmful but is sloppy. It should be added:

```python
"institutional": "institutional",  # finra_short, misc institutional signals
```

### Interaction Between P3 Fix and P4B Fix

The P3 fix (include neutral signals in convergence) changes what signals appear in `converging_signals`. This means the `_compute_confidence()` call at line 483 will now receive neutral-direction signals in its `signals` list. The Thompson-weighted geometric mean will apply their Thompson weights correctly — `finra_short` has a legitimate Thompson distribution (`finra_short` key in `thompson_distributions.json`) with 1,987 samples. This is correct behavior.

The P4B fix (per-direction dedup) is independent of the P3 fix. Both can be applied to the same file without conflict.

### File Locations for Round 2 Changes

| Fix | File | Lines Affected |
|-----|------|---------------|
| P3-A: direction logic | `mae_core/market/signal_adapters/market_data.py` | 22-24 (3 lines) |
| P3-B: neutral signal inclusion | `mae_core/market/intelligence/convergence_alerter.py` | 458-472 (`_check_direction_convergence`) |
| P4B-1: per-direction dedup dict | `mae_core/market/intelligence/convergence_alerter.py` | 199-201 (`__init__`) |
| P4B-2: threading.Lock | `mae_core/market/intelligence/convergence_alerter.py` | `__init__` + 424-437 (`check_convergence`) |
| P5B: per-domain windows | `mae_core/market/intelligence/convergence_alerter.py` | `__init__`, `_prune_old_signals`, `to_dict` |
| P5B: bootstrap wiring | `mae_core/bootstrap/market_systems.py` or `market.py` | ConvergenceAlerter instantiation |
| Bonus: add "institutional" to domain_categories | `mae_core/market/intelligence/convergence_alerter.py` | Lines 176-194 |

### Tests to Write in Round 2

For P3:
- `test_finra_short_neutral_included_in_convergence()` — verify a neutral finra_short signal contributes to domain count
- `test_neutral_only_domain_not_directional()` — verify a domain with only neutral signals doesn't anchor direction
- `test_finra_short_outcome_direction_agnostic()` — verify OutcomeCollector passes `direction=""` for neutral signals

For P4B:
- `test_bullish_bearish_dedup_independent()` — bullish firing does not suppress bearish
- `test_dedup_suppresses_same_direction_within_window()` — same direction within 4h is suppressed
- `test_dedup_allows_same_direction_after_window()` — same direction after 4h fires again
- `test_dedup_thread_safety()` — concurrent calls don't produce duplicates (use threading.Barrier)
