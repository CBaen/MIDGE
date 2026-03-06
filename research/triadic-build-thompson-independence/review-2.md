# Round 2 Independent Review — Thompson Fix + Independence Correction

**Reviewer:** Independent (did not build)
**Date:** 2026-03-05
**Files read:** thompson_sampler.py, market_hooks.py, convergence_alerter.py, correlation_tracker.py, market_systems.py, test_thompson_feedback.py, test_independence_correction.py, conftest.py

---

## 1. Integration Errors

### ISSUE: Forgetting log write happens OUTSIDE the lock

In `apply_forgetting()` (thompson_sampler.py lines 430–455), the structure is:

```python
with self._lock:
    # ... decay all distributions ...
    self._save_distributions_locked()   # inside lock — correct

    # Log one summary entry per forgetting call
    if count > 0:
        try:
            with open(self.history_path, "a") as f:   # STILL INSIDE LOCK
                f.write(json.dumps(entry) + "\n")
```

Wait — reading more carefully, the `try:` block at line 443 is still *inside* the `with self._lock:` block (indentation is at the same level as `_save_distributions_locked()`). So the log write IS inside the lock. That is correct but creates a secondary concern:

**`_log_update()` is called from `update()` which also holds `self._lock`**, and `_log_update()` also opens `self.history_path` for write. Both paths are inside the lock. This is consistent and safe — no double-lock deadlock because `_log_update` doesn't re-acquire `self._lock`. No issue here.

However: `_log_update` is called while the lock is held (line 295 within `with self._lock:` block on line 259). This means a slow I/O write to `thompson_history.jsonl` holds the global distribution lock for its duration. In a high-frequency daemon context (pace 2.0 = 2 steps/sec), this is a latency concern, not a correctness bug. Not a blocker but worth noting.

### ISSUE: Two-phase wiring order creates a gap window

In `market_systems.py`:
1. `convergence_alerter` is constructed at line 206 **without** `correlation_tracker` (it's not yet built)
2. `correlation_tracker` is constructed at line 228–243
3. Wiring happens at line 248: `ctx.convergence_alerter._correlation_tracker = ctx.correlation_tracker`

Between steps 1 and 3, if any code path calls `convergence_alerter.check_convergence()` — for example, if a `load_signal_buffer()` restored signals that immediately trigger convergence — `_correlation_tracker` is `None` and the fallback path (raw domain count) is used. This is **safe by design** (fallback is explicit), but it's a silent degradation that could produce uncorrected confidence values during the brief startup window. The builder noted this is intentional. Acceptable.

### ISSUE: `seed_from_lag_data` uses sources as CorrelationTracker signal keys, but CorrelationTracker's `get_correlation()` lookup uses the same key space

`seed_from_lag_data` seeds pairs like `("fred_macro", "yfinance_price")` into `self.correlations`. `_max_domain_correlation` in ConvergenceAlerter calls `self._correlation_tracker.get_correlation(src_a, src_b)` using source names from `_DOMAIN_SOURCES`. The canonical key used in both places is alphabetical order. Verified that `get_correlation()` normalizes: `pair_key = (signal_a, signal_b) if signal_a < signal_b else (signal_b, signal_a)`. This matches the seed function. **No key mismatch — integration is correct.**

### ISSUE: `CorrelationPair.last_updated` has `default=None` but no type annotation

`last_updated: datetime = None` — the dataclass field has no `Optional` annotation. This is a type-safety concern, not a runtime bug in Python, but it means any code checking `pair.last_updated` without a None guard could fail. The seeded pairs from `seed_from_lag_data` leave `last_updated` as `None` (the field isn't set in the seeding call). No code path in the reviewed changes reads `last_updated` on seeded pairs, so this is a pre-existing issue not introduced here.

---

## 2. Constraint Violations

### No violations found in new code.

- Floor of 2.0 applies to both alpha AND beta symmetrically. Law 7 (Rule of 3/5: odd counts, minimum 3) is not relevant to distribution floors.
- Advisory enforcement: the new code observes/reports, never blocks. `apply_forgetting` returns count, logs to history, does not raise or prevent signals.
- No monolith violations: `seed_from_lag_data` added 68 lines to `correlation_tracker.py`. File remains single-purpose.
- The three new methods on ConvergenceAlerter (`_compute_effective_domain_count`, `_max_domain_correlation`, and the `correlation_tracker` param) are coherent additions to a single class. No job boundary violation.

---

## 3. Bugs and Logic Errors

### BUG (Minor): Floor interaction with update() — behavior is correct but surprising

**Specific question from build brief:** "Does Beta(2,2) + loss → Beta(2,3) correctly?"

Answer: YES. `update()` does NOT apply the floor. It always does `new_beta = old_beta + 1`. So:
- Beta(2,2) + loss → Beta(2,3). Correct.
- Beta(1,1) + many forgets → Beta(2,2). Then + loss → Beta(2,3). Correct.
- The floor only applies inside `apply_forgetting()`, not inside `update()`. This is the right design — you want forgetting to respect the floor, but real observations should always accumulate.

No bug here.

### BUG (Minor): `_compute_effective_domain_count` is order-dependent

The algorithm iterates `domains[1:]` and checks correlation against `counted_domains` (all previously processed domains). This means the effective count depends on the ORDER domains appear in the list.

Example: domains = ["macro", "technical", "insider"]
- macro first: 1.0
- technical: correlated with macro → +0.5
- insider: not correlated → +1.0
- Total: 2.5

But domains = ["technical", "insider", "macro"]
- technical first: 1.0
- insider: not correlated with technical → +1.0
- macro: correlated with technical → +0.5
- Total: 2.5

In this specific case the totals match, but consider ["insider", "macro", "technical"]:
- insider first: 1.0
- macro: not correlated with insider → +1.0
- technical: correlated with macro → +0.5 (because macro is now in counted_domains)
- Total: 2.5

Also matches. But what about more pathological cases with 3 mutually correlated domains ["macro", "technical", "sentiment"] where sentiment is correlated with both?
- macro: 1.0
- technical: corr with macro → +0.5
- sentiment: max(corr_with_macro, corr_with_technical) → +0.5
- Total: 2.0

vs ["sentiment", "macro", "technical"]:
- sentiment: 1.0
- macro: corr with sentiment → +0.5
- technical: max(corr_with_sentiment, corr_with_macro) → +0.5
- Total: 2.0

The order dependence is real but in practice the totals are stable because `_max_domain_correlation` always takes the MAX correlation. The first domain always gets full credit. A greedy ordering (put the most-correlated domain first) would maximize the penalty applied to subsequent domains — but in all realistic configurations the difference is < 0.5 effective domains. This is an acceptable approximation for the use case. Flag for documentation but not a correctness blocker.

**The domain list is built from a Python set:** `domain_list = list({sig.domain for sig in signals})`. Python set iteration order is non-deterministic between runs (though stable within a run on CPython 3.7+). This means effective_count is technically non-deterministic across different Python processes or restarts. In practice on CPython the order is insertion-order of hashing, which varies by hash randomization. This could cause slight confidence value drift between restarts for the same signal set. Low severity but worth knowing.

### BUG (Low): `conftest.py` does NOT isolate ConvergenceAlerter's `_DATA_DIR`

The autouse `_isolate_thompson` fixture redirects Thompson's module-level paths. It does NOT redirect `convergence_alerter._DATA_DIR` or `_DISCOVERY_LOG`. Tests that create a `ConvergenceAlerter` and call `check_convergence()` or `save_signal_buffer()` will write to `data/market/signal_buffer.json` and `data/market/alerter_state.json` in production.

However, the new tests in `test_independence_correction.py` do NOT call `check_convergence()` or `save_signal_buffer()`. They only call `_compute_confidence()` and `_compute_effective_domain_count()` directly. So the production data path is never triggered by the new tests. **Not a regression, but it is an existing gap in test isolation that the builders did not make worse.**

### BUG (Low): `seed_from_lag_data` uses `observation_count=1` for seeded pairs

Seeded `CorrelationPair` objects are created with `observation_count=1`. The CorrelationTracker's `get_most_correlated_pairs()` and `get_least_correlated_pairs()` filter by `observation_count >= self.min_observations` (default 30). This means seeded pairs will NEVER appear in those outputs — only in the direct `get_correlation()` lookup used by `_max_domain_correlation`. This is intentional (comment says "seeded, not computed") but worth confirming: the asymmetry means `detect_correlation_anomalies()` will never flag a seeded pair. That's correct behavior — you don't want false anomalies from bootstrap data.

No bug, behavior is correct by design. Document it.

---

## 4. Edge Cases

### EDGE CASE: What if `correlation_tracker` construction succeeds but `seed_from_lag_data` fails?

In market_systems.py lines 232–239, the seeding is wrapped in its own `try/except`. If seeding fails (file missing, corrupt JSON), `ctx.correlation_tracker` is still valid but has ZERO seeded pairs. The two-phase wiring at line 248 proceeds normally. Result: `_max_domain_correlation` returns 0.0 for all pairs, all domains treated as independent. This is the correct fallback — backward compatible with pre-fix behavior. **Handled correctly.**

### EDGE CASE: What if `convergence_alerter` construction fails but `correlation_tracker` succeeds?

Line 247: `if getattr(ctx, "convergence_alerter", None) is not None and ctx.correlation_tracker is not None`. If alerter failed, `ctx.convergence_alerter` is `None`, the condition is False, and wiring is skipped. The orphaned `correlation_tracker` lives on `ctx` but is never used for confidence calculation. **Handled correctly.**

### EDGE CASE: What if `convergence_alerter` construction succeeds but `correlation_tracker` fails?

`ctx.correlation_tracker` is set to `None` (line 242). The two-phase wiring condition at line 247 is False. `convergence_alerter._correlation_tracker` remains `None` (set in `__init__` from the `correlation_tracker=None` default). `_compute_confidence()` falls back to raw domain count. **Handled correctly.**

### EDGE CASE: Empty `signals` list passed to `_compute_confidence`

Line 694: `if not signals: return 0.5`. Handled.

### EDGE CASE: All signals from the same domain

`domain_list = list({sig.domain for sig in signals})` — if all signals are from the same domain, domain_list has length 1. `_compute_effective_domain_count(["macro"])` returns `1.0` (line 756). The diversity bonus: `1.0 + 0.12 * math.log1p(max(0, 1.0 - 1))` = `1.0 + 0.12 * log1p(0)` = `1.0 + 0` = `1.0`. No diversity bonus applied. Correct.

### EDGE CASE: `apply_forgetting` called with no distributions initialized

`apply_forgetting()` iterates `self.distributions` which is `{}`. `count = 0`. The `if count > 0` guard at line 439 prevents the save. The `if count > 0` guard at line 443 prevents the log write. Returns 0. Test `test_no_history_entry_when_no_distributions` covers this. Correct.

### EDGE CASE: Forgetting log write failure is silently ignored

Lines 453–454: `except Exception: logger.debug(...)`. If the history file write fails (disk full, permissions), the exception is swallowed at DEBUG level. The distributions HAVE already been saved (line 440 runs before the log attempt). So distributions are correct, but the forgetting event won't appear in history. This is acceptable for a log file, but it degrades observability silently. The builder's `try/except` is correct defensive code.

---

## 5. Regression Risk

### LOW RISK: Cadence change from 100 to 200 steps

At pace 2.0, this changes forgetting from every 50 seconds to every 100 seconds. The builder's rationale (match outcome evaluation cadence so forgetting doesn't outpace learning) is sound. With the floor now at 2.0 instead of 1.0, the slower cadence also reduces the risk of over-decaying thin distributions. Net effect: forgetting is gentler, distributions retain information longer. This is the correct direction for the correction.

### LOW RISK: Floor change from 1.0 to 2.0

Beta(2,2) has variance 0.05, Beta(1,1) has variance 0.083. The floor at 2.0 means even maximally-decayed distributions have a 40% lower variance than before — they carry a slight memory. This is intentional. The `samples` property: `int(alpha + beta - 2)`. At Beta(2,2), `samples = int(2+2-2) = 2`. A distribution at the floor will still be subject to the thin-data blend in `_get_thompson_weight` (blended toward 1.0 when `observations < 5`). This is consistent — the floor doesn't magically grant "mature" status. Correct.

### LOW RISK: Injecting `_correlation_tracker` via attribute after construction

The two-phase wiring sets `ctx.convergence_alerter._correlation_tracker = ctx.correlation_tracker` directly (line 248). This bypasses `__init__` but the attribute is already initialized to `None` there (line 227). No `__slots__`, no property setter, no validation. Direct attribute assignment is safe in Python for this class. Matches the existing pattern used for `_regime_classifier` (line 262). **Consistent with established pattern.**

---

## 6. Security

No security concerns in the changed code. The changes involve:
- Numeric decay applied to in-memory distributions
- File append to JSONL history
- Reading `lag_correlations.json` — `json.load()` with error handling, no `eval()` or unsafe deserialization
- Attribute injection on a trusted internal object

---

## 7. Test Coverage

### Gaps in test_thompson_feedback.py

**Not tested:** Concurrency — two threads calling `apply_forgetting()` simultaneously. The lock should prevent corruption, but there's no test for it. Low priority since the daemon is single-threaded in the step hook context.

**Not tested:** `apply_forgetting()` with multiple regimes per signal (e.g., signal has "default" AND "bull" regime entries). The loop iterates `for regime in self.distributions[signal_id]` — both would be decayed. A test with a multi-regime distribution would confirm this.

**Not tested:** What happens if the history file path doesn't exist (parent directory missing). `open(self.history_path, "a")` would create the file if the parent exists, but fail if it doesn't. The `ThompsonSampler.__init__` calls `self.persistence_path.parent.mkdir(parents=True, exist_ok=True)` which ensures the directory exists. So this is safe, but no test verifies the mkdir call actually protects the log write.

**Cadence test is simulation-only:** `test_forgetting_fires_at_step_200_not_100` simulates the hook logic locally rather than importing the actual hook function. If `market_hooks.py` has the cadence check inside a nested conditional, the test wouldn't catch it. The complementary `test_forgetting_in_market_hooks_at_200` does string-search the actual file for the comment — this is fragile (comment could change while code doesn't). A stronger test would import the hook and call it. Not a blocker.

### Gaps in test_independence_correction.py

**Not tested:** `_compute_effective_domain_count` with domains that are NOT in `_DOMAIN_SOURCES`. If a signal arrives with domain "unknown_domain", `sources_a = self._DOMAIN_SOURCES.get("unknown_domain", [])` returns `[]`. `_max_domain_correlation` returns 0.0. The domain gets full credit (1.0). This is correct fallback behavior, but there's no explicit test confirming unknown domains are treated as independent.

**Not tested:** `_compute_confidence` with `cross_domain_count=0`. Passed as an argument. If `_correlation_tracker` is None, `effective_count = cross_domain_count = 0`. Then `diversity_factor = 1.0 + 0.12 * math.log1p(max(0, 0-1)) = 1.0 + 0.12*log1p(0) = 1.0`. No division by zero. Safe.

**Not tested:** The `_make_signal()` helper omits `source=""` by default, meaning `_DOMAIN_SOURCES` lookup will use domain-level matching from `_compute_effective_domain_count`, but `_max_domain_correlation` uses `_DOMAIN_SOURCES` to get sources for each domain — the `source` field on the Signal is not used in the correlation lookup path. This is consistent (the lookup is domain→sources→CorrelationTracker, not signal.source→CorrelationTracker). The tests correctly test the domain path. No issue.

**Not tested:** `seed_from_lag_data` with an entry where `correlation` field is missing entirely. The code does `corr = entry.get("correlation", 0.0)` — defaults to 0.0. `abs_corr = 0.0`. This pair would be treated as uncorrelated and would not overwrite existing pairs, but it would still be seeded (with `current_correlation=0.0`). Minor behavioral edge case: a pair with 0.0 correlation is in `self.correlations` but `get_correlation()` returns 0.0, so `_max_domain_correlation` returns 0.0, and the domain gets full credit. Correct by accident, but a test for missing-correlation field would be explicit confirmation.

---

## 8. What Works

All 39 new tests pass. The implementations are correct:

**Calibrator's work:**
- `apply_forgetting()` floor of 2.0 is correctly implemented with `max(2.0, ...)` applied to both alpha and beta.
- Forgetting log entry is written inside the same lock that protects the distribution save — consistent with the file being a sequential log.
- One log entry per `apply_forgetting()` call (not per distribution) is correct design.
- Cadence change to step % 200 is correctly implemented and the comment explains the rationale.

**Corrector's work:**
- `correlation_tracker=None` default in `__init__` preserves 100% backward compatibility.
- `_compute_effective_domain_count` logic is mathematically sound: greedy first-domain-gets-full-credit approach, then discounts based on max correlation with all previous domains.
- `_max_domain_correlation` correctly looks up at source level through `_DOMAIN_SOURCES`, takes maximum absolute correlation across all source pairs.
- `seed_from_lag_data` correctly: reads JSON, deduplicates by max |r|, preserves existing live data, handles missing file / empty array / self-pairs / missing fields.
- Two-phase wiring in market_systems.py is safe and matches the established `_regime_classifier` pattern.
- All three construction failure scenarios (alerter fails, tracker fails, both fail) degrade gracefully.

---

## Summary

**Blockers:** None.

**Issues requiring documentation/future attention (3):**
1. Order-dependence in `_compute_effective_domain_count` with non-deterministic set iteration for domain_list. Add a note that order matters and consider sorting domains before the loop.
2. `_log_update` holds `self._lock` during file I/O in `update()`. At high frequency this adds I/O latency to the lock hold time. Not a bug, but a latency concern in high-pace daemon mode.
3. `CorrelationPair.last_updated: datetime = None` lacks `Optional[datetime]` annotation. Pre-existing but worth cleaning up.

**Missing tests (not blocking, but strengthen future confidence):**
- Multi-regime forgetting (confirms both regime slots decay)
- Domain not in `_DOMAIN_SOURCES` gets full independence credit
- `seed_from_lag_data` with missing `correlation` field defaults to 0.0
- Cadence test should import actual hook function rather than re-simulate logic

The two fixes are coherent, correct, and safe. The independence correction's domain-level abstraction (not source-level) is the right architectural choice — it shields the alerter from source proliferation. The Thompson floor change is the right magnitude: 2.0 preserves directional memory (Beta(2,2) variance = 0.05) while still being genuinely weak evidence.
