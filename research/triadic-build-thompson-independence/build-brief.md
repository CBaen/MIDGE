# Build Brief: Thompson Feedback Fix + Independence Correction

## Date: 2026-03-05
## Project: MIDGE
## Source: Phase 0 Measurements (research/phase0-measurements.md) + Expedition Synthesis (research/expedition-competitive-edge/synthesis.md)

### Goal

Fix the two most critical bugs blocking MIDGE's ability to learn and make accurate predictions:

1. **Thompson feedback loop**: The Bayesian learning engine was working (distributions reached alpha=1389) but forgetting erased everything because it runs 2x faster than outcome evaluation. Fix the decay so learning outpaces forgetting.

2. **Independence math**: The convergence alerter treats all domain combinations as equally independent. Known correlations (macro+technical at r=0.73) inflate stacking confidence. Connect the existing CorrelationTracker to the confidence formula.

### Build Tasks

**Round 1 (parallel — independent file domains):**

#### Builder A: Thompson Feedback Fix
1. **Fix forgetting/learning cadence mismatch** — In `market_hooks.py`, the forgetting fires at `step % 100 == 0` (line 593). Outcome evaluation fires at `_outcome_cadence=200` in `sensing_hook.py` (line 399). Options:
   - (a) Align cadences: forgetting every 200 steps (match evaluation), OR
   - (b) Adaptive decay: reduce decay rate when `total_updates < 50` for a distribution, OR
   - (c) Higher floor: raise minimum from 1.0 to something that preserves learned signal (e.g., floor = max(1.0, 0.1 * peak_alpha))
   - **Recommendation: (a) + raise floor to 2.0.** Simplest fix. Forgetting every 200 steps halves the decay rate. Floor of 2.0 preserves non-uniform state.

2. **Log forgetting events** — `apply_forgetting()` in `thompson_sampler.py` (lines 411-440) calls `_save_distributions_locked()` but does NOT write to `thompson_history.jsonl`. Add a summary log entry when forgetting runs, so the history reflects actual state. Keep it compact — one entry per forgetting event, not per distribution.

3. **Clean data: remove 10 ghost predictions** — `predictions.jsonl` has 10 old-format records with no `source` field that are stuck forever. Remove them during a data cleanup pass (read, filter, rewrite).

4. **Write tests** — Verify: forgetting doesn't erase distributions below floor, forgetting logs to history, cadence alignment works.

#### Builder B: Independence Correction
5. **Inject CorrelationTracker into ConvergenceAlerter** — Add `correlation_tracker` as optional constructor parameter (backward-compatible). In `market_systems.py` bootstrap wiring, pass the existing `correlation_tracker` instance.

6. **Compute effective domain count** — In `_compute_confidence()` at line 685 of `convergence_alerter.py`, replace raw `cross_domain_count` with an effective count that discounts correlated pairs. Algorithm:
   - For each unique domain pair in the contributing signals, look up correlation from CorrelationTracker
   - If `|correlation| > 0.5`: count that pair as 0.5 domains instead of 1.0
   - If `|correlation| > 0.3`: count as 0.7
   - If no data or `|correlation| <= 0.3`: count as 1.0 (full independence)
   - `effective_domain_count = sum of per-pair contributions`
   - Use effective count in diversity_factor formula

7. **Persist correlation data** — CorrelationTracker currently stores data in memory only. Add persistence to `data/market/correlation_state.json` so correlations survive daemon restarts. Load on init, save periodically.

8. **Write tests** — Verify: correlated domains produce lower confidence than uncorrelated, CorrelationTracker injection is backward-compatible (None works), persistence round-trips correctly.

### Team Size: 2 builders + 2 reviewers = 4 agents

**Why 2 builders:** Two cleanly separated file domains with zero overlap. Thompson fix touches `thompson_sampler.py`, `market_hooks.py`, `sensing_hook.py`, `predictions.jsonl`. Independence fix touches `convergence_alerter.py`, `correlation_tracker.py`, `market_systems.py`. No shared files.

**Why 2 reviewers:** Minimum required by skill protocol. These fixes touch the two most critical subsystems (Bayesian learning + convergence confidence), warranting independent adversarial review.

### Builder Assignments

| Builder | Domain | Files Owned | Tasks |
|---------|--------|-------------|-------|
| **Calibrator** (Thompson Fix) | Learning engine | `thompson_sampler.py`, `market_hooks.py`, `sensing_hook.py`, `data/market/predictions.jsonl`, `tests/test_thompson_feedback.py` (NEW) | Tasks 1-4 |
| **Corrector** (Independence Fix) | Confidence engine | `convergence_alerter.py`, `correlation_tracker.py`, `market_systems.py`, `tests/test_independence_correction.py` (NEW) | Tasks 5-8 |

### Round Structure

**Single round.** Both builders work in parallel — their file domains do not overlap. No dependencies between tasks.

### Project Constraints

1. **Mae's Mathematical Laws** — All 8 laws apply. Especially:
   - Law 1 (No Bare Dyads): CorrelationTracker injection must be witnessed (ConnectionRegistry)
   - Law 7 (Rule of 3/5): Minimum 3 validators for any new process
2. **Zero regression policy** — `python -m pytest tests/ -v` must pass after changes
3. **Advisory enforcement** — Triads observe/report, never block
4. **No monoliths** — One job per file, flag files over 500 lines
5. **Existing test isolation** — `conftest.py` autouse fixture isolates all tests from production data

### Verification Plan

```bash
# After build
python -m pytest tests/ -v                          # Zero regressions
python -m pytest tests/test_thompson_feedback.py -v  # New Thompson tests
python -m pytest tests/test_independence_correction.py -v  # New independence tests

# Smoke test
python main.py --agents 3 --steps 100              # Quick verification

# Manual verification
python -c "
import json
with open('data/market/thompson_distributions.json') as f:
    dists = json.load(f)
uniform = sum(1 for d in dists.values() if isinstance(d, dict) and d.get('alpha', 1) == 1.0 and d.get('beta', 1) == 1.0)
print(f'Uniform distributions: {uniform}/{len(dists)}')
"
```
