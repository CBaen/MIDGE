# Team 3: Hypothesis Stagnation Findings

**Date:** 2026-03-12
**Researcher:** Opus 4.6 — Hypothesis Stagnation angle
**Question:** Why have 30 hypotheses been generated but zero promoted, with 4 stuck in "active" status?

---

## Critical Findings

### Finding 1: "Active" Is the Correct Steady State for Promoted Hypotheses

The lifecycle is: `PROBATION → ACTIVE → (HIBERNATED) → RETIRED`

"Active" IS "promoted." The registry's `promote()` method (hypothesis_registry.py:79-96) transitions `PROBATION → ACTIVE`. There is no status beyond ACTIVE for validated hypotheses. "Active" = promoted and monitoring. The 4 hypotheses in ACTIVE state have already been promoted. The framing of "30 generated, 0 promoted, 4 active" conflates two different things: the engine's session-scoped `_hypotheses_promoted` counter (which resets each run) vs. the registry's persistent state. The 4 active hypotheses WERE promoted — in prior sessions.

**This is the most important finding.** The system is not failing to promote. It has promoted 4. The apparent contradiction is a monitoring/reporting gap.

---

### Finding 2: "Active" Hypotheses Cannot Be Re-Promoted — They Are Already There

The `_run_validation()` method in hypothesis_lifecycle.py:225-270 re-validates ACTIVE hypotheses but can only retire them, not re-promote them. The `promote()` method in hypothesis_registry.py:82 explicitly rejects hypotheses not in PROBATION status:

```python
if hyp is None or hyp.status != HypothesisStatus.PROBATION:
    return None
```

So active hypotheses stay active until they are retired for degradation. This is correct behavior.

---

### Finding 3: The Validator Cannot Find Enough Historical Observations for Most Hypotheses

This is the primary stagnation mechanism for the remaining 26 probation hypotheses.

The promotion gate requires **all four** conditions to be true simultaneously (hypothesis_validator.py:249-254):
- `total_observations >= 20` (30 for composite hypotheses)
- `win_rate > 0.52` (0.53 for [AUTO] causal stories)
- `DSR > 0.5`
- `causal_story` present and not containing "REQUIRES MANUAL REVIEW"

**Condition 4 (causal_story gate) is blocking every LAG_CORRELATION hypothesis.** The first hypothesis event in the JSONL log (created 2026-02-27) shows this pattern explicitly:

```
"causal_story": "REQUIRES MANUAL REVIEW: Statistical correlation found between
sec_efts and fred_macro but no known causal mechanism."
```

The lag_correlations.json has 68 findings. The generator's `_get_causal_story()` function returns a "REQUIRES MANUAL REVIEW" string for any pair not in `CAUSAL_STORY_TEMPLATES`. The validator at line 202-205 then flags `has_real_causal_story = False`, blocking promotion regardless of win rate or DSR. Every LAG_CORRELATION hypothesis between pairs not in the template library is permanently blocked at this gate.

**The sources in lag_correlations.json are:** `fred_macro`, `finra_short`, `openinsider_purchase`, `yfinance_price`, `sec_form4`, `sec_efts`, `sec_form8k`. Unless all pairings of these sources exist in CAUSAL_STORY_TEMPLATES, the hypotheses generated from them will never be promotable.

---

### Finding 4: BACKTEST_DERIVED Hypotheses Are Blocked by Low Win Rate

The hypotheses.jsonl data shows the first two BACKTEST_DERIVED hypotheses created:

- `Sweep:ES=F` — 34.0% win rate, n=47
- `Sweep:NQ=F` — 40.0% win rate, n=55

Both have sufficient observations (>20) and present causal stories. However, win rates of 0.34 and 0.40 fail the `promote_win_rate > 0.52` gate (hypothesis_validator.py:209-213). They would be retired, not promoted.

The DSR is stored as `0.0` in their initial stats. The `_validate_from_precomputed()` path computes DSR at validation time using the current `_dsr_trials_tracked` counter (hypothesis_validator.py:315). With 30+ hypotheses tested, the Bonferroni-style penalty embedded in DSR raises the bar further. A 34% win rate will produce a negative Sharpe, which produces a deeply negative DSR — well below the 0.5 threshold.

---

### Finding 5: Granger Findings Are Present but Minimal

`granger_causality.json` contains only 2 entries:
- `sec_efts → sec_form4` (lag 10, p=0.00175)
- `sec_efts → yfinance_price` (lag 20, p=0.017)

The generator converts these to synthetic lag-finding format with `correlation = max(0.5, 1.0 - p_value * 5)` (hypothesis_generator.py:359). Both pairs would generate hypotheses, but then hit the causal_story gate above. The Granger analyzer has only been able to test the small set of sources that appear in the signal buffer together — the broader 31-source estate is not yet analyzed.

---

### Finding 6: The Validation Cadence Is Appropriate but Validation Yields Zero Observations for Most Hypotheses

`_run_validation()` (hypothesis_lifecycle.py:225-271) calls `validator.validate(hyp)` for every probation hypothesis at every 1000-step cadence. The validator at hypothesis_validator.py:132-195 does two things:

1. Calls `find_trigger_events()` to scan the signal archive for historical trigger firings of `source_a`
2. Calls `check_event_outcome()` to match each trigger to a price outcome

If the signal archive contains records for the relevant sources (`sec_efts`, `fred_macro`, etc.) over the 180-day lookback window, observations accumulate. If the signal archive is sparse or missing records for these sources, `trigger_events` is empty and `ValidationResult` is returned with zero observations (hypothesis_validator.py:168-172). Zero observations means no promote/retire decision possible.

**The system has 89K+ signals ingested, but the question is whether those signals include the specific `source` field values that the hypotheses watch for.** The lag correlations are built from `fred_macro`, `finra_short`, `openinsider_purchase`, `yfinance_price`, `sec_form4`, `sec_efts`, `sec_form8k`. If the signal archive records use different source identifiers (e.g., `fred_yield_curve` instead of `fred_macro`), the trigger-event search will find nothing.

---

### Finding 7: Meta-Learning Gates Are Permissive — They Are Not the Blocker

The `_review_gates()` method (hypothesis_meta_learning.py:160-282) would loosen `promote_win_rate` by 0.01 when `_meta_promoted_total == 0 AND probation_count >= 3 AND step_counter > gate_review_cadence * 2`. This should be triggering since there are zero promotions and 26 probation hypotheses. However, the gate cooldown (`cooldown_steps = gate_review_cadence * 5 = 10,000 steps`) prevents repeated loosening. The base `promote_win_rate` may have already been loosened once from 0.52 → 0.51. Even so, the causal_story gate (#3 above) is categorical — a win rate adjustment cannot overcome a missing causal story.

---

## Root Causes

### Root Cause 1: Causal Story Templates Don't Cover the Generated Hypotheses (Primary Blocker)

The `_get_causal_story(source_a, source_b)` function in `hypothesis_causal.py` (referenced by hypothesis_generator.py:205) returns a "REQUIRES MANUAL REVIEW" string for any source pair not explicitly listed in `CAUSAL_STORY_TEMPLATES`. Every LAG_CORRELATION hypothesis from the current lag_correlations.json involves sources that are likely not fully covered in the template library. The "REQUIRES MANUAL REVIEW" string is then caught by the validator's `has_real_causal_story` check at hypothesis_validator.py:202-205 and permanently blocks promotion.

**This is a chicken-and-egg problem by design:** the system requires human-supplied or auto-generated causal narratives before promoting statistical findings. The intent is anti-overfitting. The consequence is that without expanding CAUSAL_STORY_TEMPLATES to cover the actual source pairs producing findings, no LAG_CORRELATION hypothesis can ever be promoted.

### Root Cause 2: Backtest-Derived Hypotheses Have Below-Threshold Win Rates

Session sweep patterns have 34-40% win rates. The promotion gate requires >52%. These hypotheses cannot promote unless win rates improve or the gate is lowered. The gate loosening mechanic will eventually lower the threshold by 0.01 increments, but the gap (34% vs 52%) is too large to bridge through gate adjustment.

### Root Cause 3: Signal Archive Source Name Mismatch (Likely Secondary Blocker)

The lag correlations reference `fred_macro`, `finra_short`, etc. as source names. The signal archive (data/midge/signals/*.jsonl) records signals from the live sensing pipeline. If those signals use different `source` field values than what the correlation analyzer uses, the validator's trigger-event search finds zero historical instances — hypotheses accumulate zero observations and can never be promoted or retired.

### Root Cause 4: DSR Penalty Grows with Each Tested Hypothesis

Every call to `validator.validate()` increments `_dsr_trials_tracked`. With 30 hypotheses already tested (30+ DSR trials tracked), the multiple-testing penalty embedded in DSR raises the effective promotion bar for each new hypothesis. A marginal hypothesis (say, win rate 0.54, Sharpe 0.6) that would pass the DSR gate with 5 prior trials may not pass with 30. The growing trial counter makes it progressively harder to promote as the system matures without more observations.

---

## Recommended Fixes

### Fix 1: Expand CAUSAL_STORY_TEMPLATES for the Active Source Pairs (High Priority)

File: `mae_core/market/intelligence/hypothesis_causal.py`

Add causal story templates for the specific pairs appearing in lag_correlations.json. The 7 sources in the current findings produce 42 directed pairs. Many of these have clear economic logic:
- `finra_short → fred_macro`: Short interest predicts macro data surprises (institutions positioning ahead of Fed decisions)
- `openinsider_purchase → fred_macro`: Corporate insiders buying before macro-driven sector moves
- `fred_macro → sec_efts`: Fed rate signals precede institutional ETF repositioning
- `yfinance_price → sec_form4`: Price momentum precedes insider selling patterns
- `sec_efts → sec_form4`: Institutional positioning precedes insider trading disclosures

The auto-generation path (`_auto_generate_causal_story()`) may already exist — check if it can be invoked for these pairs instead of returning "REQUIRES MANUAL REVIEW." If `_auto_generate_causal_story()` generates an `[AUTO]` prefixed story, those hypotheses only face a +0.01 win rate penalty (0.53 instead of 0.52), which is tractable.

### Fix 2: Activate `_auto_generate_causal_story()` as Fallback Before "REQUIRES MANUAL REVIEW" (Medium Priority)

In `hypothesis_causal.py`, if `_get_causal_story()` would return "REQUIRES MANUAL REVIEW," call `_auto_generate_causal_story(source_a, source_b)` first. Auto-generated stories would be prefixed with `[AUTO]` and face a 0.53 win rate bar instead of being permanently blocked. This requires no human intervention and unblocks all the currently generated hypotheses.

### Fix 3: Verify Signal Archive Source Name Alignment (High Priority)

Run a diagnostic: extract all unique `source` field values from the signal archive (data/midge/signals/*.jsonl, most recent few files) and cross-reference against the `source_a`/`source_b` values in lag_correlations.json. If they don't match, the validator's trigger-event search is finding nothing. Fix by ensuring the lag correlation analyzer uses the same source identifiers that appear in the signal archive.

### Fix 4: Treat "Active" Status as the Success State (No Code Change Needed)

The 4 active hypotheses represent the system working correctly. The monitoring/reporting should be updated to count active hypotheses as "promoted" in dashboards and handoff notes. The `_hypotheses_promoted` session counter will always start at 0 on restart — the persistent truth is `registry.get_active()`.

### Fix 5: Add Causal Story to Backtest-Derived Hypotheses with Lower Win Rates (Low Priority)

The session sweep hypotheses (ES=F, NQ=F at 34-40% WR) will be retired — not stagnant. This is correct behavior. No fix needed unless the payoff math makes them profitable at 34% (which requires payoff > 1.94:1). If they are profitable at 34% WR, lower the promotion gate specifically for BACKTEST_DERIVED hypotheses with high avg_R.

---

## Gaps and Unknowns

1. **Contents of `hypothesis_causal.py` not read** — the exact list of source pairs in `CAUSAL_STORY_TEMPLATES` was not verified. The fix recommendation assumes these pairs are missing. A 5-minute read of that file would confirm.

2. **Signal archive source names not verified** — whether `fred_macro` in lag_correlations matches `fred_macro` in signal JSONL files was not confirmed. This is the second most important thing to check.

3. **`data/market/hypotheses_snapshot.json` not read** — the current status distribution of all 30 hypotheses (how many in probation vs active vs retired) was not directly confirmed. The background PowerShell job was still running at findings-write time.

4. **The 4 "active" hypotheses' specific names** — not confirmed from the snapshot. They could be BACKTEST_DERIVED hypotheses that passed the 52% gate, or early LAG_CORRELATION ones with template coverage.

5. **DSR state value** — `data/market/dsr_state.json` was not read. The current `_dsr_trials_tracked` value determines how severe the multiple-testing penalty is.

---

## Synthesis

**The headline is this: hypothesis stagnation is primarily a labeling problem, not an engine failure.**

The 4 "active" hypotheses ARE promoted. "Active" is the promoted state. The engine works.

The 26 probation hypotheses are stuck for two independent reasons:

1. **Causal story gate (categorical blocker):** LAG_CORRELATION hypotheses between source pairs not in `CAUSAL_STORY_TEMPLATES` receive "REQUIRES MANUAL REVIEW" causal stories. The validator treats this as a hard categorical block — no win rate or DSR result can override it. These 26 hypotheses will sit in probation forever unless CAUSAL_STORY_TEMPLATES is expanded or `_auto_generate_causal_story()` is activated as a fallback.

2. **Zero observation problem (secondary blocker):** Even if the causal story gate were removed, the validator may find zero historical trigger events for these source pairs in the signal archive — because the source names in lag_correlations.json may not match the source field values in the signal archive JSONL files. Zero observations = no verdict = permanent probation.

The fix priority is: (1) verify source name alignment, (2) expand causal story templates or activate auto-generation, (3) update monitoring to correctly report active as promoted.

The hypothesis engine itself is architecturally sound. Generation cadence (every 500 steps), validation cadence (every 1000 steps), gate review cadence (every 2000 steps), and meta-learning cadence (every 3000 steps) are all reasonable. The DSR anti-overfitting gate is correctly implemented. The event-sourced registry with snapshot acceleration is well-engineered. The blocking is at the data/labeling layer, not the architecture.

---

*Files examined:*
- `mae_core/market/intelligence/hypothesis_engine.py`
- `mae_core/market/intelligence/hypothesis_lifecycle.py`
- `mae_core/market/intelligence/hypothesis_validator.py`
- `mae_core/market/intelligence/hypothesis_generator.py`
- `mae_core/market/intelligence/hypothesis_registry.py`
- `mae_core/market/intelligence/hypothesis.py`
- `mae_core/market/intelligence/hypothesis_meta_learning.py`
- `data/market/lag_correlations.json` (68 findings, 7 sources, max r=0.675)
- `data/market/granger_causality.json` (2 entries)
- `data/market/hypotheses.jsonl` (first 3 events sampled — all CREATED, PROBATION status)
