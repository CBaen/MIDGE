# MIDGE Handoff

## What Happened

### Pattern Archaeologist — Full Excavation In Progress (2026-03-04)

**ACTIVE BACKGROUND PROCESS:** Full excavation running via `populate_library.py --reset --batch-size 50`. Using Polygon.io paid API + optimized caching. Check status with:
```bash
grep "Batch [0-9]" excavation_optimized.txt | tail -5
```

If interrupted, resume with: `python populate_library.py --batch-size 50` (progress auto-saved).

**Polygon.io Bulk Fetcher (2026-03-04):**
- `mae_core/market/archaeology/polygon_bulk_fetcher.py` — NEW. Drop-in replacement for PriceFetcher. Uses per-ticker aggregate endpoint. 1 API call per symbol returns all daily bars (up to 50K). ~0.5s per symbol vs yfinance's ~3min.
- `populate_library.py` — Updated: auto-detects MASSIVE_API_KEY, loads `.env` file, `--source polygon|yfinance|auto` flag.
- Polygon Starter plan ($29/mo) activated by Guiding Light.

**Computation Optimizations (2026-03-04):**
- `historical_fetcher.py`: Added `preload_archive()` — loads all 900+ signal archive JSONL files into memory once at startup. Eliminates redundant disk I/O.
- `historical_fetcher.py`: Added `_get_ta_cached()` — computes RSI/MACD/Bollinger/Volume once per symbol for full price history, caches result. Previously recomputed identically for EVERY dig site (~50x per symbol).
- `historical_fetcher.py`: `clear_cache()` now only clears per-symbol TA cache, preserves pre-loaded archive.
- Combined speedup: ~7x faster than yfinance baseline (85s per 100 symbols vs 6min).

**Performance benchmarks (100 symbols):**
| Approach | Time | Projected 3,237 |
|----------|------|-----------------|
| yfinance (no cache) | ~6 min | ~10 hours |
| Polygon (no cache) | ~3.3 min | ~1.8 hours |
| Polygon + TA cache + preloaded archive | 85s | ~46 min |

### Pattern Archaeologist v2 — Symbol-Agnostic Template Engine (2026-03-04)

Reworked from narrow (specific-source matching) to universal (domain-level templates). This is the "convergence of convergences" — when multiple independent historical patterns stack on the same ticker, that's the 95%+ signal.

**Core concept shift:** Fingerprints are instances. **PatternTemplates** are the abstraction — grouped by `direction + domain_signature`. A template like "bullish: insider+macro+technical" accumulates instances across NVDA, AAPL, MSFT. Cross-symbol validation (3+ symbols) boosts confidence.

**New/rewritten files:**
- `mae_core/market/archaeology/fingerprint.py` — `PatternTemplate`, `TemplateInstance` dataclasses. Template auto-ID is deterministic hash of direction+domain_signature.
- `mae_core/market/archaeology/excavator.py` — Takes `HistoricalDataFetcher`, excavates from all 29 sources via domain mapping. `_SOURCE_DOMAIN_MAP` converts sources → 11 domains.
- `mae_core/market/archaeology/historical_fetcher.py` — 3-tier data retrieval + TA caching + archive pre-loading.
- `mae_core/market/archaeology/pattern_library.py` — Template-based storage. `query_similar()` matches by domain overlap. Stores both fingerprints and templates in separate JSONL files.
- `mae_core/market/archaeology/pattern_watcher.py` — Domain-level independence checks. Stacking tiers (low/medium/high). `PatternActivation` carries `template`, `matched_domains`, `missing_domains`.
- `mae_core/market/archaeology/excavation_daemon.py` — Step hook (every 5000 steps). Persistent progress tracking.
- `mae_core/market/archaeology/polygon_bulk_fetcher.py` — Polygon.io paid API bulk fetcher.
- `mae_core/bootstrap/market_systems.py` — Wires PatternLibrary + PatternWatcher + ExcavationDaemon into Layer 33.

**Feedback loop (2026-03-04):**
- `outcome_collector.py`: `register_pattern_stack()` registers stacks as predictions. `_on_outcome_graded()` callback updates template win/loss stats via PatternLibrary.
- `outcome_tracker.py`: `on_outcome` callback (fires after each graded prediction).
- `market_hooks.py`: Wires pattern stack registration + pattern library feedback.
- `OUTCOME_WINDOWS["pattern_stack"] = 14` — 14-day outcome window.

**Synergy detection (2026-03-04):**
- `market_hooks.py`: `ctx._cached_pattern_stacks`. When ConvergenceAlerter AND PatternWatcher both fire on same ticker+direction, emits `CH_DUAL_CONFIRMATION` with `combined_confidence = 1 - (1-conv_conf)(1-stack_conf)`.
- `channels.py`: `CH_DUAL_CONFIRMATION = "market.intel.dual_confirmation"`.

**Tests:** 82 tests (68 archaeology + 14 outcome collector). All pass. Full suite: 746 passed, 1 pre-existing flaky.

---

### From Proven Signal to Profitable System — 4 Operational Fixes (2026-03-03)

Replay analysis proved MIDGE has real statistical edge (z=4.74, p<0.0001) but can't capitalize due to operational gaps. Four work packages close the gap.

**WP1 — Thompson Persistence Protection:** `tests/conftest.py` autouse fixture monkeypatches all data paths to tmp_path. No test can touch production data.

**WP2 — ctx.outcome_collector Wiring:** `market_hooks.py` added `ctx.outcome_collector = outcome_collector`. Closes combo Thompson feedback loop.

**WP3 — Confidence-Gated Paper Trading:** `learning_config.py` added `paper_trade_min_confidence: 0.45`. Combo filter blocks combos with historical WR < 25%.

**WP4 — Magnitude-Aware Replay Grading:** `replay_history.py` MFE/MAE tracking, expectancy, Sharpe ratio, return percentiles.

---

## Stats

- **146 systems** (92 core + 54 market), **4,292 tests**, **157 holons**, **425 connections**
- **97 market files** (29 API + 12 edge + 27 intelligence + 8 signal_adapters + 8 archaeology + 1 polygon fetcher + 12 root)
- **33-layer bootstrap**, **14 mixins** on MycelialAgent

## What's Next

- **Check excavation completion:** `grep "Excavation finished" excavation_optimized.txt` or `python populate_library.py --dry-run`
- **After excavation:** Library will have ~150K+ fingerprints and ~20+ domain-level templates across 3,237 symbols. Run MIDGE daemon (`python main.py --daemon --agents 6 --steps 500 --pace 2.0`) and PatternWatcher will match live signals against templates.
- **Future optimization:** Concurrent symbol processing (ProcessPoolExecutor) for another ~3-4x speedup.
- **Options flow via Unusual Whales** ($35/mo API — needs Guiding Light approval on spend).

## Verification

```bash
python -m pytest tests/ -q              # 746 pass (1 pre-existing flaky)
python main.py --agents 3 --steps 30    # Smoke test
python populate_library.py --dry-run     # Check excavation status
```
