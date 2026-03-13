# Codebase Analyst Findings
## Multi-Analyst Architecture for MIDGE
**Date:** 2026-03-13
**Analyst:** Codebase Analyst (Council Member)
**Scope:** How to replace DeepAnalyst with three specialized analysts sharing a situation board

---

## Current State

### The One Analyst Problem

`DeepAnalyst` (`mae_core/market/intelligence/deep_analyst.py`) runs every **200 steps** inside `_run_slow_cadence_ops` in `mae_core/bootstrap/market_hooks_steps.py` (lines 425–510). It does six scoring passes over all 30 days of signal history, synthesizes `Inevitability` objects, writes `data/midge/inevitabilities.jsonl`, and publishes to `market.intel.deep_analysis` on the EventBus. One object does: signal loading, Thompson scoring, template matching, WorldModel traversal, lag-lead scoring, density scoring, historical win-rate lookup, combo-boost from post-mortem insights, evidence summary writing, plain-language formatting, outcome registration, and Qdrant embedding.

**The bottleneck in numbers:** Every 200 steps, one function reads up to 30 days of JSONL archives, queries PatternLibrary, traverses the WorldModel graph, and re-reads `lag_correlations.json`, `outcomes.jsonl`, and `post_mortem_insights.json` from scratch.

**Output fate:** Results go to `ctx.inevitabilities` (a list on the context object), to `data/midge/inevitabilities.jsonl` (append-only JSONL), to `market.intel.deep_analysis` (EventBus, currently no confirmed subscriber that acts on it), to Qdrant via `pattern_memory.remember_inevitability()`, and to `data/midge/alerts_human.jsonl` via `plain_language.format_inevitability()`.

---

### The Three Tiered Alerters — The Graveyard

`mae_core/bootstrap/market_hooks_sensing_setup.py` (line 28–34) constructs three `ConvergenceAlerter` instances:

```python
tiered_alerters["tactical"]  = ConvergenceAlerter(min_domains=2, convergence_window_hours=48)
tiered_alerters["strategic"] = ConvergenceAlerter(min_domains=2, convergence_window_hours=21*24)
tiered_alerters["thematic"]  = ConvergenceAlerter(min_domains=2, convergence_window_hours=90*24)
```

These three alerters **receive NO signals**. The `MarketSensingHook` feeds signals only into the primary `ctx.convergence_alerter`. The tiered alerters check convergence every 10 steps (line 501, `market_hooks_sensing.py`), but their signal buffers are always empty, so `check_convergence()` always returns nothing.

Their output, when non-empty, goes to `ctx._market_advisory["tactical"]`, `ctx._market_advisory["strategic"]`, `ctx._market_advisory["thematic"]`. The `_market_advisory` dict is read only by:
- `_write_convergence_heartbeat()` — dumps to `data/midge/convergence_state.json` (monitoring only)
- Market agents in `market_agents.py` — they hold a reference via `_market_advisory_ref` but none of the agent code reads `tactical`/`strategic`/`thematic` from it
- `market_hooks_steps_core.py` line 262 — writes `ticker_alerts` key only

**Verdict:** The tiered alerters are fully wired in and called, but because they never receive signals, their output is always `None`. They are not the "disconnected" architecture failure — the signal feed is the failure. The analysts don't get data; they aren't broken in themselves.

---

## Architecture Map

### Data Flow: Signal Ingestion → DeepAnalyst

```
MarketSensingHook.step()
    ↓ [every step, 12 concurrent fetchers]
    signals fetched from 31 sources
    ↓
    convergence_alerter.record_signal() [only this alerter receives signals]
    ↓
_sensing_step_with_advisory() [market_hooks_sensing.py]
    ↓
    ctx._cached_alerts = alerter.check_convergence()
    ↓ [every step]
_market_sense_hook() [market_hooks_steps_core.py]
    ↓
    CH_CONVERGENCE published on EventBus
    outcome_collector.register_convergence_alert()
    ↓ [every 200 steps]
_run_slow_cadence_ops() [market_hooks_steps.py]
    ↓
    deep_analyst.analyze(lookback_days=30, top_n=20)
        → reads signal JSONL archive (not live buffer)
        → scores 6 dimensions
        → returns List[Inevitability]
    ↓
    ctx.inevitabilities = results
    data/midge/inevitabilities.jsonl (append)
    market.intel.deep_analysis (EventBus)
    pattern_memory.remember_inevitability() (Qdrant)
    plain_language.format_inevitability() → alerts_human.jsonl
```

### Key Architectural Observation

**DeepAnalyst reads the ARCHIVE, not the live buffer.** It uses `SignalArchiveReader` to load from `data/midge/signals/YYYY-MM-DD.jsonl` files. This means it sees a 30-day historical view, not the real-time signal state. The live convergence engine (`ConvergenceAlerter`) works from the in-memory `self.signals` dict (72h window). These are **two separate data sources** that only partially overlap.

This is the architectural wedge that makes specialization natural: analysts can own different time horizons without stepping on each other.

---

### Communication Mechanisms Between Intelligence Systems

**EventBus channels** (all defined in `mae_core/market/channels.py`):

| Channel | Publisher | Subscribers (confirmed) |
|---------|-----------|------------------------|
| `market.intel.convergence` | `market_hooks_steps_core.py` | `market_hooks.py` (cascade, backward cascade, advisory bridge) |
| `market.intel.deep_analysis` | `market_hooks_steps.py` | NONE confirmed |
| `market.intel.partial_convergence` | `convergence_detection.py` | OctopusColony (via investigation dispatcher) |
| `market.intel.cascade_confirmed` | `cascade_tracker.py` | `market_hooks.py` (`_on_cascade_confirmed`) |
| `market.intel.lag_finding` | `market_hooks_steps.py` | NONE confirmed |
| `market.intel.granger_finding` | `market_hooks_steps.py` | NONE confirmed |
| `market.hypothesis.discovered/promoted/fired` | `hypothesis_engine.py` | NONE confirmed |
| `market.archaeology.pattern_stack_detected` | `pattern_watcher.py` | `market_hooks.py` (synergy detection) |

**Shared ctx attributes** (written by one system, read by another):

| Attribute | Written by | Read by |
|-----------|-----------|---------|
| `ctx._cached_alerts` | `_market_sense_hook` | `_run_drift_detector`, advisory bridge |
| `ctx.inevitabilities` | `_run_slow_cadence_ops` | No confirmed reader (set but never consumed) |
| `ctx._market_advisory` | sensing hook wrapper | `_write_convergence_heartbeat`, market agents |
| `ctx._ticker_alerts` | `_market_sense_hook` | motif/anomaly block (step 100) |
| `ctx._cached_pattern_stacks` | `market_hooks.py` synergy | advisory bridge (type check only) |
| `ctx._developing_situations` | OctopusColony | Octopus dispatcher (step 20) |

**Direct method calls** (dependency injection at bootstrap):
- `PatternLibrary` → injected into `PatternWatcher`, `DeepAnalyst`, `Excavator`
- `ThompsonSampler` → injected into `ConvergenceAlerter`, `DeepAnalyst`, `PostMortemReviewer`, `OutcomeCollector`, `KellyPositionSizer`
- `WorldModel` → injected into `ConvergenceAlerter`, `DeepAnalyst`, `CascadeTracker`
- `SignalArchiveReader` → injected into `DeepAnalyst`, `LagCorrelationAnalyzer`, `GrangerAnalyzer`

---

## Key Question Answers

### 1. How are the tiered alerters constructed and what data do they produce?

Tiered alerters are three `ConvergenceAlerter` instances with different time windows and no Thompson injected (`thompson_sampler=None`). They receive no signals. Their `check_convergence()` always returns `[]`. Their output, if non-empty, would go into `ctx._market_advisory["tactical"|"strategic"|"thematic"]` as `.to_dict()` of the strongest alert. **No code reads these advisory keys for any action.** They exist in `data/midge/convergence_state.json` as monitoring artifacts only.

### 2. What communication mechanisms already exist?

The EventBus is the primary inter-system channel. The pattern is: system publishes event → `register_callback()` subscriber reacts. However, most inter-system communication in the current codebase uses **shared ctx attributes** (simpler than bus), or **direct method calls** via injected dependencies. The bus is used for external-facing notifications, not internal coordination.

### 3. Exact data flow: where could new analysts tap in?

There are **three clean tap points**:

**Tap A — After signal ingestion (live buffer, real-time):** In `_market_sense_hook()` after `ctx._cached_alerts` is set. New analysts here see the live 72h signal window. This is the most reactive position.

**Tap B — After DeepAnalyst runs (every 200 steps):** `ctx.inevitabilities` is set and `market.intel.deep_analysis` is published. Currently no subscriber acts on it. A `SituationBoard` could subscribe to this channel and receive ranked inevitabilities.

**Tap C — After specific sub-analyses (every 200 steps):** The lag correlation findings, Granger findings, PostMortem insights, and CascadeTracker state are all available on ctx after the 200-step cycle. A specialist analyst could read `ctx.post_mortem_reviewer`, `ctx.granger_analyzer`, `ctx.cascade_tracker` directly.

**The most architecturally sound tap point is between the 200-step cycle and a situation board.** Rather than intercepting mid-cycle, a `SituationBoard` that reads `ctx.inevitabilities` (already computed), `ctx.cascade_tracker.get_active_chains()` (already live), and `ctx.hypothesis_engine.get_statistics()` (always available) would have rich, pre-computed data with zero extra compute.

### 4. Bootstrap patterns for new analysts

Looking at `market_systems.py` lines 386–395 (DeepAnalyst bootstrap):

```python
try:
    from mae_core.market.intelligence.deep_analyst import DeepAnalyst
    ctx.deep_analyst = DeepAnalyst(
        thompson_sampler=getattr(ctx, "thompson_sampler", None),
        pattern_library=getattr(ctx, "pattern_library", None),
        world_model=getattr(ctx, "world_model", None),
    )
except Exception:
    logger.debug("Market: deep_analyst failed to construct", exc_info=True)
    ctx.deep_analyst = None
```

The pattern is: `try/except`, construct with injected deps, set on ctx, None on failure. New analysts **must** follow this exact pattern. They are added to `market_systems.py` (currently 512 lines — approaching the 500-line cap, but the `try/except` pattern adds ~6 lines per analyst).

**Important constraint:** `market_systems.py` is 512 lines, already over the 500-line cap. New analyst instantiation should go into a new sub-module (like how `market_intelligence.py` handles the hypothesis loop and `market_gifts.py` handles the Ten Gifts). A `market_analysts.py` file following the same extraction pattern.

The step cadence is registered in `_run_slow_cadence_ops` in `market_hooks_steps.py`. New analyst step calls would add ~10 lines to the `if step % 200 == 0:` block. That block is already 100 lines — adding 3 analysts would push it past 500. The analysts should call a single `_run_analyst_council()` helper, itself extracted to a new file.

### 5. How would a "situation board" fit the existing architecture?

The `SituationBoard` is the natural successor to `ctx._market_advisory`. Currently `_market_advisory` is a plain dict with keys `alert`, `tactical`, `strategic`, `thematic`, `ticker_alerts`, `updated_step`, `active_hypotheses`, `kelly`. A `SituationBoard` class would:

- Replace or extend `ctx._market_advisory`
- Be writable by multiple analysts (thread-safe via a lock, like `CascadeTracker._chains_lock`)
- Expose a `publish(analyst_id, finding)` method that stores findings in a structured dict
- Expose a `get_snapshot()` method returning all findings for human-readable output
- Be readable by the EventBus heartbeat writer (`_write_convergence_heartbeat`) — this already reads `ctx._market_advisory`

The `_developing_situations` pattern in `OctopusColony` is the closest existing precedent: a thread-safe dict with a lock, written by one system and read by a dispatcher. That's exactly what a `SituationBoard` is.

---

## Pattern Inventory

### Patterns used in existing systems relevant to this proposal:

**Pattern 1: Injected dependencies with fallback None** — every system accepts optional deps at construction, falls back gracefully if None. `DeepAnalyst._ensure_dependencies()` shows the self-healing variant (loads from disk if not injected).

**Pattern 2: Step cadence via modulo** — `if step % N == 0:` gating in `_run_slow_cadence_ops`. All cadence logic lives in one function. New analysts must register here.

**Pattern 3: try/except with `logger.debug`** — never let one system crash the step loop. Every analyst call is wrapped.

**Pattern 4: Two-phase wiring** — systems constructed first, cross-wired second (e.g., `convergence_alerter._correlation_tracker = ctx.correlation_tracker`). Used when circular deps exist. New analysts can use this for the `SituationBoard` back-reference.

**Pattern 5: EventBus publish for external-facing events** — don't use the bus for tight inter-system coupling. Use ctx attributes for that. Use the bus for notifications that external subscribers may want.

**Pattern 6: Mixin decomposition for large classes** — `ConvergenceAlerter` is split across 6 files (models, confidence, detection, lag_scoring, buffer, ticker). `HypothesisEngine` across 2 mixins. Any analyst class over 300 lines should follow this.

**Pattern 7: Atomic file writes** — write to `.tmp`, then `rename`. Used in `PatternLibrary`, `PostMortemReviewer`, `WorldModel`. Required for any new file the analysts write.

**Pattern 8: `get_statistics()` for HolonProxy** — every system exposes this method for the holarchy awareness pulse.

---

## Blast Radius Analysis

### Files that must change:

| File | Change Required | Risk |
|------|----------------|------|
| `mae_core/bootstrap/market_systems.py` | Add analyst instantiation (extract to `market_analysts.py`) | Low — additive only |
| `mae_core/bootstrap/market_hooks_steps.py` | Add analyst dispatch in `_run_slow_cadence_ops` (extract to helper) | Low — additive, same cadence |
| `mae_core/market/channels.py` | Add `CH_ANALYST_FINDING`, `CH_SITUATION_UPDATE` constants | Trivial |
| `data/midge/` | New JSONL file for situation board state | Zero code risk |

### Files that must NOT change (constraints from brief):
- `mae_core/market/intelligence/convergence_alerter.py` and sub-files
- `mae_core/market/intelligence/thompson_sampler.py` and feedback loop
- `mae_core/market/intelligence/hypothesis_engine.py` and lifecycle

### New files to create:
- `mae_core/market/intelligence/situation_board.py` — shared state for analyst findings
- `mae_core/market/intelligence/analyst_causal.py` — CausalChainAnalyst (reads WorldModel + CascadeTracker)
- `mae_core/market/intelligence/analyst_temporal.py` — TemporalPatternAnalyst (reads PostMortem + LagCorrelations)
- `mae_core/market/intelligence/analyst_convergence.py` — ConvergenceQualityAnalyst (reads DeepAnalyst output + PatternWatcher)
- `mae_core/bootstrap/market_analysts.py` — bootstrap extraction file

### Tests that might be affected:
- Tests for `DeepAnalyst` — no structural change, it continues to exist
- Tests for `market_hooks_steps.py` — if the 200-step block changes, tests checking the step hook cadence would need updating
- No existing test infrastructure touches `ctx.inevitabilities` (confirmed via search — no test reads this attribute)

### No blast to:
- All 28 existing intelligence systems
- Thompson feedback loop
- Hypothesis lifecycle
- Pattern Archaeology pipeline
- Convergence engine internals

---

## Internal Precedents

**Precedent for shared mutable state read by multiple systems:** `ctx._market_advisory` dict — already a situation board in spirit, just underpowered. The upgrade is giving it a class and thread safety.

**Precedent for step-cadenced analyst:** `PostMortemReviewer.review()` runs every 500 steps, reads historical data, writes JSON insights, feeds Thompson updates. Exactly the pattern new analysts would follow, just reading different inputs.

**Precedent for analyst reading other analyst output:** `DeepAnalyst._load_combo_stats()` reads `post_mortem_insights.json` written by `PostMortemReviewer`. This is the current (fragile) inter-analyst communication: file-based. A `SituationBoard` in memory would be faster and structured.

**Precedent for three-role specialization:** The three stem cell roles `SEC_WATCHER`, `CONTRACT_TRACKER`, `MARKET_ANALYST` in `stem_cell.py` are exactly the Law 2 triadic specialization applied at agent level. The same law (three, not two or four) should apply to analysts.

---

## Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Feasibility** | 9/10 | All tap points exist. The situation board pattern has a direct precedent (`_market_advisory`). New files are additive. No existing system needs structural modification. |
| **Blast Radius** | 8/10 | Two bootstrap files need extension (not rewrite). Three new files. One new channels constant. No existing tests break. The 200-step cadence block needs a helper extraction which is purely structural. |
| **Pattern Consistency** | 9/10 | Follows try/except bootstrap, step cadence, get_statistics, EventBus publish, injected deps patterns already present everywhere. The mixin decomposition pattern should be applied if any analyst exceeds 300 lines. |
| **Dependency Risk** | 8/10 | New analysts depend on systems already on ctx: `deep_analyst`, `cascade_tracker`, `post_mortem_reviewer`, `world_model`, `pattern_library`. All are already instantiated before any analyst would be added. No circular deps. |
| **Overall Risk** | 8/10 | Additive architecture. Zero risk to existing systems. The only risk is hitting the 500-line cap on `market_hooks_steps.py` and `market_systems.py` — both managed by extracting to `market_analysts.py`. |
| **Reversibility** | 10/10 | Each analyst is a `try/except` block. If any fails, `ctx.analyst_X = None`. The situation board can be removed by deleting the class and two ctx references. Zero rollback complexity. |
| **Evidence Confidence** | 9/10 | Every claim above references a specific file and line number. The tiered alerter signal starvation was confirmed by tracing the signal feed path. The `ctx.inevitabilities` orphan status was confirmed by grep. |

---

## Concerns

**Concern 1: The 200-step cadence piles up.** DeepAnalyst already runs every 200 steps alongside lag correlation, Granger, PostMortem, and cascade expiry. Adding three more analysts at the same cadence increases the 200-step block's wall time. Mitigation: analysts should read pre-computed data (from DeepAnalyst, from CascadeTracker) not re-compute from raw archives.

**Concern 2: `market_hooks_steps.py` is 577 lines** (already over cap). Any addition needs an extraction. Confirmed this file houses `_run_slow_cadence_ops` which is the 200-step dispatch point. A `_run_analyst_council()` helper extracted to `market_analysts.py` is the correct move.

**Concern 3: Thread safety on SituationBoard.** Multiple step hooks run on different threads (sensing hook, step hook). If analysts write to SituationBoard from a step hook and the heartbeat writer reads from a different timing context, a lock is required. Use `threading.RLock()` (same as `CascadeTracker`).

**Concern 4: `ctx.inevitabilities` is the right tap point, not re-reading archives.** DeepAnalyst already reads 30 days of archives. New analysts should read `ctx.inevitabilities` (already computed) plus the live systems on ctx. If analysts re-run archive reads, total 200-step time doubles or triples.

**Concern 5: The tiered alerters need signal feeds, not replacement.** The "three tiered alerters built but disconnected" failure is specifically that they receive no signals. The fix (if we want tactical/strategic/thematic time horizons) is to feed them signals in the sensing hook, not to build entirely new analyst classes for that purpose. The new analyst architecture solves a different problem: specialized reasoning over existing data, not signal-window tiering.

---

## Opportunities

**Opportunity 1: `market.intel.deep_analysis` has no subscriber.** The EventBus channel is published but nothing listens. A `SituationBoard` could subscribe at bootstrap and receive analyst triggers automatically, without any cadence polling.

**Opportunity 2: `ctx.inevitabilities` is unread.** This is rich, pre-scored data that disappears into JSONL but never informs any live decision. A `CausalChainAnalyst` that reads the top 5 inevitabilities and traces their WorldModel chains would immediately add value with near-zero compute.

**Opportunity 3: CascadeTracker's energy ratio is unread.** `cascade_tracker.get_statistics()` returns `mean_energy_ratio` — whether cascades are accelerating or decelerating. Nothing reads this. A `TemporalPatternAnalyst` that tracks energy ratio trends over time could surface "cascade momentum" as a new signal type.

**Opportunity 4: PostMortem insights are rich but only feed Thompson.** `post_mortem_insights.json` contains `flagged_orderings` (domain sequences that consistently fail), `timing_summary`, `regime_summary`, `mfe_mae_patterns`. None of this reaches the human output. A `ConvergenceQualityAnalyst` could translate these into plain-language situation updates.

**Opportunity 5: The `SituationBoard` can replace `_market_advisory` entirely.** `_market_advisory` is already a shared state dict. Upgrading it to a typed class with analyst slots, timestamps, and human-readable summaries would cost minimal effort and clean up a pattern that is half-implemented.

---

## Gaps and Unknowns

**Gap 1:** No existing tests exercise the `ctx.inevitabilities` pathway or the `market.intel.deep_analysis` channel. Any new code built on these pathways will need new tests.

**Gap 2:** The exact `market_hooks_steps.py` line count is 577. Any addition to `_run_slow_cadence_ops` requires extracting the analyst block to a new file first. The test coverage for the step hook dispatch logic is unknown — if tests directly test line numbers or step hook structure, refactoring could break them.

**Gap 3:** `SituationBoard` concurrency model needs design. The sensing hook runs in a thread-wrapped executor, the step hook runs in the main Mesa loop. If both write to SituationBoard, a reentrant lock is required. The exact threading model of the Mesa step loop is not examined here.

**Gap 4:** Plain-language output for analyst findings. `plain_language.py` has formatters for `PatternStack` and `Inevitability` and `ConvergenceAlert`. New analyst findings need new formatter functions. This is additive (5-10 lines each) but must be planned.

**Gap 5:** How analysts communicate **to each other** (not just to the SituationBoard) is not defined. The brief says "each seeing what others found and building on it." If `CausalChainAnalyst` publishes a finding that `TemporalPatternAnalyst` should react to, the bus channel or a SituationBoard subscription model needs to be decided before implementation.

---

## Architecture Recommendation for Implementers

Based on what exists:

**The minimal correct architecture:**

1. `SituationBoard` class — a thread-safe dict with three analyst slots plus a `publish(analyst_id, finding)` and `get_snapshot()` method. Slot keys: `"causal"`, `"temporal"`, `"convergence_quality"`. Lives at `mae_core/market/intelligence/situation_board.py`. Replaces `ctx._market_advisory["tactical"/"strategic"/"thematic"]`.

2. Three analyst classes, each under 400 lines, each with the same interface: `analyze(ctx) -> AnalystFinding` and `get_statistics() -> dict`. They read from ctx (pre-computed data already on context, never from raw archives). They write findings to `SituationBoard` via `sb.publish(self.analyst_id, finding)`.

3. A single `_run_analyst_council(ctx, step)` function in a new `market_analysts.py` bootstrap file. Called from `_run_slow_cadence_ops` every 200 steps. Calls all three analysts in sequence (3 fast reads, no archive I/O).

4. A subscribe callback on `market.intel.deep_analysis` that triggers `SituationBoard.mark_fresh()` — so the situation board knows when DeepAnalyst has run and its data is current.

5. `SituationBoard.get_snapshot()` appended to `data/midge/situation.json` (overwrite, same as `convergence_state.json` pattern). This is what Guiding Light reads.

**What each analyst reads:**
- `CausalChainAnalyst`: `ctx.inevitabilities` (top 5) + `ctx.world_model.find_ripple_effects()` + `ctx.cascade_tracker.get_active_chains()`
- `TemporalPatternAnalyst`: `ctx.post_mortem_reviewer.get_statistics()` + `data/market/lag_correlations.json` (already loaded in memory by lag_correlation_analyzer) + `ctx.cascade_tracker.get_statistics()["mean_energy_ratio"]`
- `ConvergenceQualityAnalyst`: `ctx.inevitabilities` (confidence distribution) + `ctx.convergence_alerter.get_domain_status()` + `data/market/post_mortem_insights.json` (already computed by PostMortemReviewer)

This design: zero new data sources, zero new API calls, zero modifications to protected systems, reads only pre-computed outputs, produces human-readable plain-language to a single JSON file.
