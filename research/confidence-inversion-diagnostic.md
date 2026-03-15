# MIDGE Confidence Inversion Diagnostic
**Date:** 2026-03-15
**Investigator:** Subagent analysis of convergence pipeline
**Trigger:** Post-mortem claim that high-confidence (0.80+) predictions have 0% win rate

---

## Executive Summary

The confidence inversion is real but the sample size is critically small. The post-mortem finding is based on **4 predictions total** (the entire graded prediction history). Three structural bugs cause confidence to be inversely correlated with actual quality in the convergence engine. A fourth finding — that ALL 116 high-confidence discovery log entries are test fixtures — means MIDGE has **never produced a real high-confidence convergence alert in production**.

---

## Part 1: Where Does the "0.80+ = 0% Win Rate" Claim Come From?

The post-mortem reviewed **23 outcomes** (from outcomes.jsonl). Of these, only **4 have confidence values**:

| Confidence | Outcome | Signals |
|-----------|---------|---------|
| 0.60 | FALSE | technical_rsi, politician_sell |
| 0.75 | TRUE | technical_macd, insider_form4 |
| 0.767 | TRUE | cci, stochastic, williams_r |
| 0.855 | FALSE | rsi, stochastic, bollinger, williams_r, cci |

**The 0.855 prediction is the entire basis for the "0.80+ = 0% win rate" claim.** One data point. This is statistically meaningless.

Additionally: the 0.855 prediction for LMT (bearish) used 5 TA indicators that are ALL from the same underlying price data. It is not a convergence alert — it came from a different system entirely (likely the TA-level signal evaluator, not ConvergenceAlerter).

---

## Part 2: The Three Structural Mechanisms That Would Cause Inversion

Even though the sample is too small to prove inversion empirically, the code contains three bugs that **would produce** confidence inversion at scale if not fixed.

### Bug 1: Geometric Mean Poisoning by Low-Quality Signals

**Location:** `convergence_confidence.py`, `_compute_confidence()`, lines 236–259

The geometric mean is correct when all inputs are independent. But MIDGE mixes high-quality signals (insider, macro at confidence=0.70) with low-quality or hardcoded signals (sentiment=0.45, institutional_synthesis=0.15, government=0.50, cascade=0.50). Adding a weak signal **always reduces** the geometric mean, even though it is supposed to represent more evidence.

**Demonstration (from actual buffer values):**

```
3 signals @ 0.70 (3 domains): confidence = 0.792
+ 1 signal @ 0.45 (sentiment): confidence = 0.731   [DECREASED]
+ 1 signal @ 0.20:              confidence = 0.597   [DECREASED FURTHER]
+ 1 signal @ 0.15 (inst_synth): confidence = 0.555   [NEARLY HALVED]
```

**Result in production data (from discovery_log.jsonl, 4756 real alerts):**

| Domain Count | N Alerts | Avg Confidence |
|-------------|----------|----------------|
| 3 | 2,918 | 0.604 |
| 4 | 364 | 0.523 |
| 5 | 865 | 0.315 |
| 6 | 498 | 0.484 |
| 7 | 111 | 0.476 |

**More domains = lower confidence.** The diversity bonus (max +25%) cannot overcome the geometric mean penalty from low-quality signals. The engine is rewarding scarcity of evidence, not convergence.

The most extreme example: `events+fundamentals+insider+macro+sentiment+technical` (6-domain combo) consistently produces confidence < 0.25. The culprit is `institutional_synthesis` which has a hardcoded confidence value of **0.15** for all 446 signals in the buffer. One signal with confidence=0.15 dragged a 5-signal 0.70 set down to confidence < 0.25.

### Bug 2: Hardcoded Confidence Values Are Not Empirical

**Location:** Signal generation code across multiple adapters and APIs

Signal confidence values are largely **hardcoded**, not derived from any measured reliability. From the current signal buffer:

| Domain | Confidence Values | Assessment |
|--------|------------------|------------|
| cascade | [0.5] | Hardcoded |
| contracts | [0.5] | Hardcoded |
| crypto | [0.5] | Hardcoded |
| government | [0.5] | Hardcoded |
| institutional | [0.5] | Hardcoded |
| institutional_synthesis | [0.15] | Hardcoded, critically low |
| sentiment | [0.45] | Hardcoded |
| macro | [0.5, 0.7] | Partly hardcoded |
| insider | Variable 0.0–1.0 | Derived from signal data |
| technical | Variable 0.45–0.80 | Derived from indicators |

When `institutional_synthesis` (hardcoded at 0.15) appears in a convergence, it functions as a confidence poison pill. The geometric mean of [0.70, 0.70, 0.70, 0.15] is 0.476 — worse than random.

**The Thomson weighting does NOT fix this.** Thompson weights are in range [0.5, 1.5] and multiply the log-contribution of each signal. A signal with confidence=0.15 has log(0.15) = -1.9 vs log(0.70) = -0.36. Even a Thompson weight of 0.5 (minimum) only reduces the contribution by half — the signal still dominates negatively.

### Bug 3: Combo Thompson Has Negative Feedback Loop on Real Combos

**Location:** `convergence_confidence.py`, `_apply_confidence_modifiers()`, lines 410–417

Combo Thompson distributions for real combos all have mean < 0.5 — meaning they **reduce** confidence. The best real-world combo:

| Combo | Mean | Multiplier | Effect |
|-------|------|------------|--------|
| combo:a+b+c | 0.989 | 1.489 | **TEST FIXTURE ONLY** |
| combo:events+macro+price+sentiment+technical | 0.657 | 1.157 | n=28 (thin) |
| combo:events+macro+price | 0.408 | 0.908 | Reduces by 9.2% |
| combo:events+insider+macro | 0.105 | 0.605 | Reduces by 39.5% |
| combo:events+institutional+macro+price | 0.123 | 0.623 | Reduces by 37.7% |
| combo:events+insider+institutional+macro+price | 0.085 | 0.585 | Reduces by 41.5% |

The combos that fire most often are the ones being penalized most heavily. This is theoretically correct — they historically lose more than they win — but it creates a paradox: the combos that have the most evidence are being trained to report low confidence, while rare combos with little data report higher confidence (because their Thompson hasn't been trained down yet).

---

## Part 3: The High-Confidence Alert Fiction

**This is the most important finding:**

The discovery_log.jsonl contains **116 alerts with confidence >= 0.80**. Every single one is a test fixture.

- **42 alerts**: domain combo `[a, b, c]` — test data with synthetic signals `s1, s2, s3`
- **44 alerts**: `events+macro+price` with signals `sig_events, sig_macro, sig_price` — integration test fixtures
- **30 alerts**: `insider+macro+technical` with signals `s1, s2, s3` — integration test fixtures

**MIDGE has zero production convergence alerts with confidence >= 0.80. The maximum observed production confidence is 0.792.**

The `combo:a+b+c` Thompson distribution has been trained by these test fixtures to mean=0.989 (677 samples). This fictional 1.489x multiplier is sitting in the live thompson_distributions.json waiting to be applied to any real combo that happens to be named "a+b+c". This is not a live threat (no real combo is named that) but it shows the test-production contamination risk.

---

## Part 4: Independence Correction — Does It Work?

**Location:** `convergence_confidence.py`, `_compute_effective_domain_count()`, lines 261–285

The Phase 0 finding was that macro+technical have r=0.73. The independence correction discounts correlated domain pairs by giving them +0.5 credit instead of +1.0.

**Status:** The correction is correctly implemented but **rarely activates**, because it requires `self._correlation_tracker` to have measured correlations via CorrelationTracker. The CorrelationTracker needs time (many steps) and live data to build up correlation measurements between source pairs. At startup and in early operation, it has no data, so `_max_domain_correlation()` returns 0.0 for all pairs, and all domains get full +1.0 credit — the correction is silently bypassed.

From `lag_correlations.json`:

```json
// This file contains Granger causality data (directional lag correlations)
// The convergence independence correction uses _correlation_tracker.get_correlation()
// which is a DIFFERENT system (cross-signal Pearson r, not Granger F-test)
```

The two correlation systems are not connected. Granger data is never used for independence correction.

---

## Part 5: What Actually Produces Confidence 0.80+ (When It Occurs)

The formula requires signal.confidence values averaging around 0.70 for 3 domains to cross 0.80:

```
3 signals @ 0.70: geo_mean = 0.700, diversity(3) = 1.132, base = 0.792
After combo:events+macro+price modifier (0.908): final = 0.719
```

To reach 0.80+ requires either:
1. Signal confidence > 0.75 AND no combo Thompson penalty (combo not in distributions yet)
2. A combo Thompson boost (only `combo:events+macro+price+sentiment+technical` qualifies with n=28)
3. The test fixture path (`combo:a+b+c`)

In practice, the only sources that produce confidence > 0.70 in individual signals are:
- `fred_macro` via raw_data_analyst Pattern A: hardcoded confidence=0.85 (yield inversion + CPI)
- `fred_macro` via Pattern B: hardcoded confidence=0.80 (recession risk)
- `session_sweep_ifvg`: variable, can reach 0.80

---

## Root Causes Summary

| # | Bug | Location | Effect |
|---|-----|----------|--------|
| 1 | Geometric mean poisoned by low-quality signals | `_compute_confidence()` | More domains = lower confidence (inverted) |
| 2 | Signal confidence values are hardcoded, not empirical | Signal generation across all adapters | institutional_synthesis=0.15 is a poison pill |
| 3 | Combo Thompson penalizes the most-observed combos | `_apply_confidence_modifiers()` | Real combos get confidence-reduced; rare combos do not |
| 4 | Independence correction only works when CorrelationTracker has data | `_compute_effective_domain_count()` | Correlated domains (macro+technical, r=0.73) counted as independent |
| 5 | Test fixture Thompson data in production distributions | `combo:a+b+c` in thompson_distributions.json | Fictional 1.489x multiplier ready to fire |

---

## What Is NOT Broken

- The geometric mean formula itself is mathematically correct for independent signals.
- The Thompson weight scheme (0.5 + mean → [0.5, 1.5]) is sound in principle.
- The combo Thompson mechanism is correct in theory — it learns which combos win.
- The diversity bonus saturates appropriately (max ~25%).
- The coherence multiplier (0.5 + 0.5 * coherence) correctly penalizes contradictions.

---

## Recommended Fixes (Prioritized)

**Fix 1 (Highest impact): Replace hardcoded confidence with empirical signal quality scores**
Every domain that currently uses confidence=0.5 is contributing nothing but noise to the geometric mean. Either:
- Measure each source's hit rate and use that as confidence
- Use Thompson sampler's `dist.mean` as the signal-level confidence (already available)
- Set a floor: no signal enters geometric mean with confidence < 0.4

**Fix 2: Remove institutional_synthesis from convergence input or set its floor to 0.4**
The 0.15 hardcoded confidence is actively destroying confidence calculations. It is the primary cause of 5-6 domain alerts producing confidence < 0.25.

**Fix 3: Switch geometric mean to a minimum-of-medians or signal-floor approach**
Instead of penalizing the geo mean for every additional signal, consider: if N signals all agree directionally with mean confidence C, the combination confidence should be >= C, not < C. The current formula violates this.

**Fix 4: Remove test fixture data from thompson_distributions.json**
Delete `combo:a+b+c`, `combo:a+b+c+d+e`, `combo:idem+test+combo`, `combo:mean+test+key` from the production distributions file. These are contaminating the learned distributions.

**Fix 5: Connect Granger causality to independence correction**
The Granger data in `granger_causality.json` knows which source pairs have directional lag relationships. These should feed the `_max_domain_correlation()` lookup, not wait for the slow CorrelationTracker to build up Pearson r values.

---

## The Bottom Line

MIDGE's confidence engine does not discriminate winners from losers because:

1. **Signal confidence is fictional** — most sources emit hardcoded values unrelated to their actual hit rate
2. **More convergence = lower confidence** — the geometric mean is poisoned by low-quality signals that inflate domain count without adding real information
3. **The combos that fire most are the ones most penalized** — Thompson correctly learns they lose, but the signals that produced those losses still carry the same hardcoded confidence

The confidence number that MIDGE reports is primarily a function of which domains happened to fire (domain count and their hardcoded confidence floors) and which combo Thompson distributions have been trained, not of actual predictive power.

The "inversion" the post-mortem found — while based on only 4 data points — would be expected to hold at scale based on this analysis.
