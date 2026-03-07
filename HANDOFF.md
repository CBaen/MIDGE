# MIDGE Handoff

## What Happened

### Template Persistence Fix + Performance Optimization (2026-03-06)

**Critical bug found and fixed:** 222K fingerprints existed but 0 templates — PatternWatcher had nothing to match against. Three compounding bugs:

1. **Crash-unsafe write:** Templates used `open("w")` (truncate-on-open). Any crash or restart with empty `self._templates` zeroed the file. Fingerprints survived because they use `open("a")` (append).
2. **Per-fingerprint persistence:** `_persist_templates()` was called 222K times per excavation run (once per fingerprint via `_update_template()`). Catastrophic I/O.
3. **O(N²) rebuild:** `rebuild_templates()` did linear scan over all templates per fingerprint. `add_instance()` iterated ALL instances to compute avg.

**Fixes:**
- Atomic write: `.tmp` then rename, with Windows fallback (other process may hold file lock)
- Empty guard: Never overwrite non-empty file with empty data
- Batch persistence: `store_batch()` persists templates ONCE at the end
- O(1) key index: `_template_key_index` dict replaces linear scan in `_update_template()` and `rebuild_templates()`
- Incremental avg: `_move_pct_sum` field — O(1) per add instead of O(N) instance scan
- Instance cap: Templates keep last 200 instances (fingerprints are the archive)
- 11 new tests in `test_template_persistence.py`

**Result:** 222,916 fingerprints → 39 templates (26 cross-validated) rebuilt in 6.8s. PatternWatcher now operational.

**Files changed:**
- `mae_core/market/archaeology/pattern_library.py` — Atomic write, batch persist, key index
- `mae_core/market/archaeology/fingerprint.py` — Incremental avg, instance cap, `_move_pct_sum` field
- `tests/test_template_persistence.py` — NEW: 11 tests

### Thompson + Independence Fix (2026-03-05)

**Thompson forgetting bug:** Forgetting (0.99x every 100 steps) outran learning (outcomes every 200 steps). 81/83 distributions decayed to uniform. Fix: cadence 100→200, floor 1.0→2.0.

**Independence correction:** CorrelationTracker was NOT connected to ConvergenceAlerter. Diversity bonus used raw domain count. Fix: inject CorrelationTracker, compute effective domain count (|r|>0.5 → half credit), seed from lag_correlations.json.

**Files changed:**
- `mae_core/market/intelligence/thompson_sampler.py` — Floor 2.0, forgetting summary log
- `mae_core/bootstrap/market_hooks.py` — Cadence 200 steps
- `mae_core/market/intelligence/convergence_alerter.py` — Effective domain count, correlation injection
- `mae_core/market/intelligence/correlation_tracker.py` — `seed_from_lag_data()`
- `mae_core/bootstrap/market_systems.py` — Correlation seeding + two-phase wiring
- `tests/test_thompson_feedback.py` — NEW: 16 tests
- `tests/test_independence_correction.py` — NEW: 23 tests

### Prediction-to-Action (2026-03-05)

Three features per Guiding Light's directive:
- **Dynamic Outcome Windows**: `lag_profile_raw` accumulator, `expected_move_window_days` property
- **Plain-Language Alerts**: `plain_language.py` — zero-jargon 5-section formatter
- **Active Tracking**: `active_tracker.py` — TrackedAsset registry with status transitions

### Pattern Archaeologist v2 (2026-03-04)

Symbol-agnostic template engine. Full excavation completed 3,237 symbols via Polygon.io.

### EIA Energy Data Integration (2026-03-06)

**First real-economy domain.** All 11 prior MIDGE domains were financial-market data. EIA adds physical-world supply/demand signals that cross-reference with insider trades, congressional activity, and technical patterns.

**What it does:**
- Fetches weekly petroleum inventory (crude, gasoline, distillate), natural gas storage, and monthly crude production from EIA API v2
- Inventory BUILD = bearish (supply > demand), DRAW = bullish — inverse logic
- Strength calibrated against typical weekly change ranges
- Affected tickers mapped: XLE, XOP, USO, UNG, VLO, MPC, EQT, etc.
- 6-hour cache (weekly data updates on specific days)
- Strategic tier in convergence engine, 7-day domain window
- Full intelligence layer: Thompson key, source_reliability (0.70), energy decay rate (0.05), domain correlation tracking

**API bugs fixed (live-tested against EIA):**
- Added `data[]=value` param (EIA v2 requires explicit column selection — without it, returns metadata only)
- Gasoline/distillate facets: `SAX` → `SAE` (Ending Stocks, not Excluding SPR)
- Natgas facets: `SAX` → `SWO` (Working Gas total)
- Crude production: added `series: MCRFPUS1` facet (prevents multi-series collision)

**Live data (2026-02-27 report):** All 5 series returning — crude stocks BUILD +3,475K bbl (bearish 0.69), gasoline BUILD +801K (bearish 0.40), distillate BUILD +429K (bearish 0.21), natgas BUILD +65 Bcf (bearish 0.81), crude production +9,655K bbl/mo (bearish 0.97).

**Files changed:**
- `mae_core/market/apis/eia_client.py` — NEW: EIAClient, EnergyIndicator, 5 series + API bug fixes
- `mae_core/market/signal_adapters/market_data.py` — `from_energy_indicator()` adapter
- `mae_core/market/sensing_fetchers.py` — `fetch_eia()` function
- `mae_core/market/sensing_hook.py` — SOURCE_ROTATION (29), TIER_ROUTING, _ROTATION_TO_THOMPSON, _ABSENCE_SOURCE_DOMAINS, __init__, _fetch_source
- `mae_core/bootstrap/market_systems.py` — EIAClient instantiation + trust registration (0.95)
- `mae_core/bootstrap/market_hooks.py` — Pass eia_client to MarketSensingHook
- `mae_core/market/archaeology/pattern_library.py` — `"eia_energy": "energy"` in _SOURCE_DOMAIN_MAP
- `mae_core/market/intelligence/convergence_alerter.py` — Energy domain window + category + _SOURCE_TO_THOMPSON_KEY + _DOMAIN_SOURCES
- `mae_core/market/intelligence/learning_config.py` — source_reliability + decay_rates
- `mae_core/market/plain_language.py` — Energy domain + source translations
- `mae_core/market/signal_adapters/__init__.py` — Re-export from_energy_indicator
- `mae_core/market/signal.py` — Re-export from_energy_indicator
- `tests/test_eia_client.py` — NEW: 38 tests (34 original + 4 intelligence layer)
- `tests/test_new_source_wiring.py` — Updated rotation count 28→29
- `tests/test_integration.py` — Added eia_client to market_keys

**Requires:** `EIA_API_KEY` env var (free: https://www.eia.gov/opendata/register.php) — registered and set in `.env`

### Proven Signal → Profitable System (2026-03-03)

Four work packages closing the operational gap (Thompson isolation, combo feedback, confidence gating, MFE/MAE).

---

## Stats

- **146 systems** (92 core + 54 market), **4,484+ tests**, **157 holons**, **425 connections**
- **102 market files** (31 API + 12 edge + 27 intelligence + 8 signal_adapters + 10 archaeology + 14 root)
- **12 domains**, **30 sources** in sensing rotation
- **33-layer bootstrap**, **14 mixins** on MycelialAgent
- **222,916 fingerprints**, **39 templates** (26 cross-validated across 3+ symbols)

## Current State

- **Daemon: STOPPED.** Old daemon (PID 184380, since March 3) and excavation (PID 262480, since March 4) were killed — running pre-fix code.
- **Templates: REBUILT.** 39 templates live in `pattern_templates.jsonl`. PatternWatcher can now match live signals.
- **Thompson: FIXED.** Forgetting/learning cadence aligned. Independence correction active.
- **EIA: LIVE.** All 5 energy series returning real data. Intelligence layer fully wired (Thompson, correlation, decay).
- **Congress.gov: WIRED.** Legislative signal client integrated. 11 policy areas mapped to sector ETFs. Full intelligence layer wiring (Thompson, convergence, plain language, 51 tests).
- **Needs restart:** `python main.py --daemon --agents 6 --steps 500 --pace 2.0`

## What's Next

1. **Restart daemon on fixed code** — picks up all fixes: Thompson, independence, templates, active tracking, EIA energy, Congress.gov
2. **Monitor template feedback loop** — watch for template win/loss updates in `pattern_templates.jsonl`
3. **New real-economy domains** — USDA agriculture (free, seasonal), BDI logistics (free proxy)
4. **Pattern discovery upgrades** — Granger causality (statsmodels), transfer entropy, RMT denoising, PCMCI+
5. **Web scraping infrastructure** — autonomous pattern discovery via website crawling (research complete, stack: httpx + selectolax + trafilatura)
6. **Options flow via Unusual Whales** ($35/mo API — needs Guiding Light approval)
7. **Re-run excavation** — companion process with fixed template code + new EIA + Congress.gov domains

## Verification

```bash
python -m pytest tests/ -q              # 4433+ pass, 0 regressions
python main.py --agents 3 --steps 30    # Smoke test
python -c "from mae_core.market.archaeology.pattern_library import PatternLibrary; lib = PatternLibrary(); print(f'{lib.size} fingerprints, {lib.template_count} templates')"
```
