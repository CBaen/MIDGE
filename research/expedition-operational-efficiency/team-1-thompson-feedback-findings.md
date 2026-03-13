# Team 1 Findings: Thompson Feedback Loop
## Date: 2026-03-12

---

### Critical Findings

**The loop is architecturally intact but has two compounding failure modes that together guarantee zero learning at current cadences.**

#### Finding 1: Forgetting Gate Blocks All Decay (And Therefore All Learning Signal)

`market_hooks_steps_core.py` lines 174–201 show the Bayesian forgetting cadence is every 75 steps. The gate on line 187 is:

```python
if current_evaluated > _last_evaluated_count[0]:
    sampler.regime_aware_forget(regime)
    _last_evaluated_count[0] = current_evaluated
else:
    logger.debug("Skipping Thompson forget — no outcomes graded since last forget")
```

This gate was introduced correctly to prevent forgetting from outpacing learning. However, it has an unintended consequence: **the comment in the log says distributions are staying at prior, which means `current_evaluated` is perpetually 0.** If no outcomes are ever graded (because windows are too long — see Finding 2), the gate permanently blocks forgetting. This is not itself a bug, but it confirms the symptom: zero grading = zero updates.

#### Finding 2: Source Key Mismatch — Signals Registered Under Keys That Thompson Does Not Track

This is the primary structural break in the feedback loop.

`OutcomeCollector.register_signals()` (outcome_collector.py line 180) calls:
```python
self.tracker.record_prediction(source=sig.source, ...)
```

The `sig.source` values come from the signals as produced by fetchers. The actual source names in active rotation (from `sensing_constants.py` SOURCE_ROTATION / TIER_ROUTING) are keys like:
- `"ta_rsi"`, `"ta_macd"`, `"ta_bollinger"`, `"finnhub_realtime"`, `"crypto_coingecko"`, `"economic_calendar"`, `"fred_yields"`, `"openinsider_purchase"`, etc.

The Thompson distributions in `data/market/thompson_distributions.json` carry keys like:
- `"technical_macd"`, `"insider_form4"`, `"technical_rsi"`, `"politician_sell"`, `"rsi"`, etc.

The `_ROTATION_TO_THOMPSON` map in `sensing_constants.py` translates rotation names → Thompson keys for the **sampling** path (so convergence_alerter can weight signals correctly). But **this translation is NOT applied on the feedback path.** When OutcomeTracker calls `thompson_sampler.update(signal_id=source, ...)`, the `source` value is the raw fetcher key (e.g. `"ta_rsi"`), not the Thompson key (e.g. `"technical_rsi"`). The update lands in a brand-new distribution slot that no one ever samples from. The seeded distributions (technical_rsi, technical_macd, etc.) never receive any updates.

**Concrete path:**
1. Sensing fetcher produces signal with `source="ta_rsi"`
2. `sensing_collector.py` line 190: `outcome_collector.register_signals(signals)`
3. `outcome_collector.py` line 180: `tracker.record_prediction(source="ta_rsi", ...)`
4. After window expires: `outcome_tracker.check_pending_outcomes()` calls `thompson_sampler.update("ta_rsi", success)`
5. Thompson creates a new `"ta_rsi"` key at Beta(1,1) and updates it — the `"technical_rsi"` distribution (which convergence uses for weighting) remains untouched

#### Finding 3: Outcome Window Lengths Prevent Any Near-Term Grading

The shortest outcome windows are:
- `sec_form8k`: 5 days
- `congressional`: 14 days
- `correlation`: 21 days
- Everything else: 45–90 days

The daemon runs at `--pace 2.0` (2 seconds/step). At 75-step forgetting cadence, that is ~150 seconds between forgetting cycles. The outcome evaluator runs every 75 steps in `sensing_hook.py` (line 277). But predictions cannot be graded until their window elapses in wall-clock days, not steps. A fresh MIDGE instance will have zero mature predictions for at least 5 days. This is by design but means the first real Thompson update is at minimum 5 days away from a cold start.

#### Finding 4: Two Separate `OutcomeCollector` Instances — One Gets Thompson, One Gets ctx

`market_hooks_sensing_setup.py` (lines 36–57) creates an `OutcomeCollector` and stores it on `ctx.outcome_collector`. This instance receives the correct `ctx.thompson_sampler`.

However, `market_systems.py` lines 322–332 also constructs a bare `OutcomeTracker` (not `OutcomeCollector`) and stores it on `ctx.outcome_tracker`. This second object is redundant but harmless — it is never called for evaluation. The `outcome_collector` is the live one.

The 2026-03-09 fix (Bug 1: wrong ThompsonSampler instance) appears correctly resolved — `market_hooks_sensing_setup.py` line 39 reads `_ts = getattr(ctx, "thompson_sampler", None)` which is the canonical instance. The `id()` logging at line 46 confirms identity at boot.

#### Finding 5: Convergence Combo Keys Are Untracked in Thompson

`OutcomeCollector.register_convergence_alert()` (line 212) registers predictions under keys like `"combo:events+macro+price"`. These keys have no corresponding Thompson distribution. `OutcomeTracker.check_pending_outcomes()` will call `thompson_sampler.update("combo:events+macro+price", success)` which creates a new never-sampled slot. Combo Thompson uses a separate seeded distribution path (`seed_combo_distributions` from replay results), but the update path deposits outcomes into a differently-named key than what was seeded.

---

### Root Causes

1. **Primary (structural):** The `_ROTATION_TO_THOMPSON` translation map exists for the sampling direction but is not applied on the update/feedback direction. Outcomes update raw fetcher key names; convergence alerter samples from canonical Thompson key names. The two namespaces never meet.

2. **Secondary (temporal):** Outcome windows (5–90 days) are measured in wall-clock time. No grading can occur in the first 5 days of operation. The forgetting gate correctly suppresses decay during this window, but the side effect is that all distributions appear frozen.

3. **Tertiary (redundant object):** `ctx.outcome_tracker` (bare `OutcomeTracker`) sits unused alongside `ctx.outcome_collector` (the live `OutcomeCollector`). Not a bug, but creates confusion about which object owns the feedback loop.

---

### Recommended Fixes

#### Fix 1 (High Priority): Apply Key Translation on the Update Path

In `mae_core/market/outcome_tracker.py`, when `check_pending_outcomes()` calls `thompson_sampler.update()`, the source key must be translated through `_ROTATION_TO_THOMPSON` before the call.

Simplest approach: import the map and apply it in `OutcomeTracker.check_pending_outcomes()`:

```python
# At top of outcome_tracker.py (or passed in as a constructor arg)
from mae_core.market.sensing_constants import _ROTATION_TO_THOMPSON

# Inside check_pending_outcomes(), before calling update:
thompson_key = _ROTATION_TO_THOMPSON.get(source, source)
self.thompson_sampler.update(thompson_key, success=success, regime=regime)
```

Alternatively, store the mapping at prediction-record time: when `record_prediction(source=...)` is called, also store the translated `thompson_key` in the prediction record JSON so the grading path can use it without importing sensing_constants.

**Files to edit:**
- `mae_core/market/outcome_tracker.py` — update() call site
- OR `mae_core/market/intelligence/outcome_collector.py` — translate at `record_prediction()` call time

#### Fix 2 (Medium Priority): Add a thompson_distributions.json Audit at Boot

Add a startup log that compares `thompson_sampler.distributions.keys()` against all keys that will be written by `update()`. Any mismatch should warn loudly. This would have surfaced Fix 1 immediately.

**File to edit:** `mae_core/bootstrap/market_hooks_sensing_setup.py` — after OutcomeCollector construction, log `"Thompson keys: %s"` and `"Known rotation-to-thompson keys: %s"`.

#### Fix 3 (Low Priority): Remove Redundant `ctx.outcome_tracker`

`market_systems.py` lines 322–332 construct a bare `OutcomeTracker` that never evaluates predictions. Remove this construction to eliminate confusion about which object owns the feedback loop.

**File to edit:** `mae_core/bootstrap/market_systems.py` — remove the OutcomeTracker block (lines 321–332).

#### Fix 4 (Low Priority): Verify Combo Key Seeding Matches Update Keys

The `seed_combo_distributions()` method seeds from replay results using combo strings. Confirm the seeded key format matches what `register_convergence_alert()` produces. If `seed_combo_distributions` seeds `"combo:events+macro+price"` and `register_convergence_alert` also registers `"combo:events+macro+price"` (they appear to match by inspection), this path is correct. If not, apply the same key normalization.

---

### Gaps and Unknowns

1. **`_ROTATION_TO_THOMPSON` contents not fully confirmed.** `sensing_constants.py` was read from line 1 to 80; the map may start later. Confirm it maps e.g. `"ta_rsi"` → `"technical_rsi"` before applying Fix 1.
2. **Outcome window wall-clock measurement confirmed**, but the exact `signal_date` stored in predictions.jsonl was not verified. If `signal_date` is stored as `datetime.now()` at registration time rather than the original signal timestamp, windows start fresh on each MIDGE restart rather than from the original signal date. This could extend the effective grading delay further.
3. **Whether any predictions.jsonl records exist** from the current live run was not checked. If the file is empty or missing, the loop has never registered any predictions and the source-key mismatch is moot until registration is confirmed working.

---

### Synthesis

The Thompson feedback loop is architecturally complete and correctly wired at the coarse level: `OutcomeCollector` has the right `ThompsonSampler` instance, `sensing_collector.py` calls `register_signals()`, and `sensing_hook.py` calls `_evaluate_outcomes()` on cadence. The 2026-03-09 four-bug fix appears to have held.

The failure is a namespace gap: signals are registered under their fetcher source names (e.g. `"ta_rsi"`, `"finnhub_realtime"`, `"openinsider_purchase"`) but Thompson distributions are keyed under canonical names (e.g. `"technical_rsi"`, `"finnhub"`, `"openinsider"`). The `_ROTATION_TO_THOMPSON` translation map that bridges these namespaces on the sampling path was never applied to the feedback path. Every outcome update silently creates a new orphaned distribution slot rather than updating the seeded ones.

Fix 1 is a single-file change of approximately 3 lines. It is the highest-leverage fix in the entire MIDGE operational picture. The temporal issue (5+ day window before first grading) is by design and cannot be shortened without compromising signal validity.
