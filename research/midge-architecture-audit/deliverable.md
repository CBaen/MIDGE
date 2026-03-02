# MIDGE Architecture Audit — Collaborative Deliverable

**Triadic Team:** Lead (Signal Pipeline Architect), Alpha (Adversarial Market Practitioner), Beta (Systems Reliability Engineer)
**Date:** 2026-03-01
**Phases Completed:** Independent analysis → Cross-review → Researched revision → This deliverable

---

## Executive Summary

Three independent analysts audited MIDGE's complete architecture from signal ingestion to financial output. The audit found that MIDGE has built a sophisticated pattern detection organism with a genuine Bayesian learning loop — but **it cannot trade**. The `TradeSignal` dataclass is defined and never instantiated. Every convergence alert fires into a void.

Beyond the missing output path, the audit found **contaminated learning data** (test fixtures in production, mock outcomes with impossible returns), an **unfixed thread-safety bug** that already caused one full Thompson distribution rebuild, and a **reward signal that teaches agents the wrong objective** (TaskPool busywork valued 2x higher than market intelligence).

The triadic process also **reversed a false consensus**: all three auditors initially concluded that `finra_short` (MIDGE's most data-rich signal at 1,263 samples) was an anti-signal performing below random. Beta's base rate research proved it is a direction-agnostic volatility signal with **13-15 percentage points of genuine positive edge** over baseline. Had any single auditor's recommendation been followed, MIDGE would have lost its most validated signal.

---

## The Five Priorities

### Priority 1: Fix the Bayesian Learning Foundation
**Consensus: 3/3 — Unanimous**

The Thompson Sampler is the brain of MIDGE's learning. It is currently compromised in three ways:

**1A. Thread-safety bug (CF-1)** — `thompson_sampler.py:_log_update()` (line 272) appends to `thompson_history.jsonl` without acquiring `self._lock`. The lock exists and protects `_save_distributions()` but not the history append. This exact bug pattern previously corrupted the Thompson distributions, requiring a full rebuild from 9,462 outcomes (documented in MEMORY.md). **It is still unfixed.**
- Fix: One line — move `_log_update()` call inside `self._lock` context.
- Also: Wrap `self.distributions[signal_id][regime]` mutation (lines 241-244) inside the same lock.

**1B. Non-atomic writes (CF-2)** — Four persistence files use `Path.write_text()` (truncate+write). A crash between truncate and write produces an empty file. Files affected: `learning_config.py`, `hypothesis_engine.py`, `hypothesis_generator.py`, `step_timer.py`.
- Fix: Replace with `os.replace(tmp, path)` pattern already used correctly in `_save_distributions()`.

**1C. Data contamination (CF-3)** — Production data files contain test artifacts:
- `pair_outcomes.json`: 8 entries with keys "a|b", "a0|b0" through "a6|b6" (test fixtures)
- `outcomes.jsonl`: Records with impossible 49.93% single-day returns for AAPL (mock data)
- `predictions.jsonl`: A prediction timestamped 2027 in a 2026 system
- `learning_config.py`: `congressional = 0.75` while actual Thompson mean is 0.164
- Fix: Purge contaminated records, update config defaults, rebuild Thompson distributions.

**Why this is first:** Every confidence score, every convergence gate, every hypothesis promotion decision downstream of Thompson is only as good as the distributions. Fixing anything else first builds on a contaminated foundation.

---

### Priority 2: Build the Paper Trading Output Path
**Consensus: 3/3 — Unanimous (with sequencing from Beta)**

MIDGE detects patterns but cannot act on them. The `TradeSignal` dataclass exists at `signal.py:82-94` but is never instantiated anywhere in the codebase. The `KellyPositionSizer` computes fractions that nothing reads. The endocrine cascade converts convergence alerts into dopamine releases that widen agent exploration — but produce no trade.

**What to build:**
1. Gate in `sensing_hook.py:_collect_one()`: When convergence alert has `confidence > 0.75` AND `strength > 0.65`, instantiate `TradeSignal`
2. Write to `data/midge/paper_trades.jsonl` with: timestamp, ticker, direction, entry price (yfinance), Kelly fraction, confidence, contributing domains, dedup signal_id
3. Configure `paper_account_value` in learning_config.py (e.g., $100,000) — Kelly fraction needs a denominator
4. Wire `OutcomeCollector` to evaluate paper trade exits against real prices and compute dollar P&L
5. Build dedup gate INTO the paper trading path (signal_id check within 4-hour window) — prevents the alert storm bug from producing 20 simultaneous paper trades

**Prerequisite:** Priority 1A (Thompson lock) must be fixed first. Building P&L tracking on corrupted confidence scores teaches the wrong lessons. CF-1 is a one-line fix — the delay is hours, not weeks.

**Why this matters:** Without dollar P&L, the Thompson distributions, DSR-gated hypothesis lifecycle, and Kelly sizer are all optimizing toward directional accuracy at 5% threshold — a proxy for financial return, not financial return itself.

---

### Priority 3: Correct the finra_short Analysis + Fix Direction-Agnostic Signal Handling
**Consensus: 3/3 after revision (initially 3/3 wrong)**

**The triadic process's most important finding.** All three auditors initially concluded finra_short (35.8% win rate, 1,263 samples) was performing below random — either "anti-signal" (Alpha), "marginal edge" (Lead), or "needs base rate research" (Beta).

Beta's research proved everyone wrong:

- `outcome_tracker.py` lines 291-302: When `direction == ""`, success = stock moved 5% in EITHER direction
- Every `finra_short` record in `outcomes.jsonl` has `direction: ""`
- finra_short is **not predicting direction** — it is predicting **volatility** ("this heavily-shorted stock will move significantly")
- The correct baseline is the direction-agnostic 5% move rate (~22-23% from yfinance_price Thompson data)
- finra_short at 35.8% represents **13-15 percentage points of positive edge over baseline** on 1,265 samples

**What to fix:**
1. Verify the finra_short signal adapter preserves `direction=""` end-to-end through OutcomeCollector
2. In convergence_alerter.py, ensure direction-agnostic sources don't incorrectly reinforce directional convergence when combined with directional signals
3. Update `learning_config.py finra_short` default to ~0.36 (matching actual Thompson mean)
4. Do NOT flip, exclude, or demote finra_short — it is MIDGE's most data-rich validated signal

**Why this matters:** Had any single auditor's Phase 1 recommendation been followed, MIDGE would have lost its strongest validated signal. The triadic cross-review process caught this by forcing the base rate question.

---

### Priority 4: Fix Agent Reward Misalignment + Alert Deduplication + Static Confidence
**Consensus: 3/3 on all three sub-items**

**4A. Reward misalignment** — Market-role agents (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST, HYPOTHESIS_EXPLORER, HYPOTHESIS_VALIDATOR) are capped at 0.5 reward for market intelligence actions while TaskPool abstract tasks return up to 1.0. Over thousands of VDN Q-table training steps, agents learn that busywork is 2x more valuable than market discovery.
- Fix: For market roles, cap TaskPool exploit at 0.3 and raise market action ceiling to 0.8.
- File: `market_actions.py` reward return values.

**4B. Alert deduplication** — The dedup guard in `convergence_alerter.py:check_convergence()` lines 424-437 allows 20+ duplicate alerts within a single second (confirmed in `discovery_log.jsonl`). Root cause: fast step loop calls `check_convergence()` multiple times within one wall-clock second.
- Fix: `threading.Lock()` around check-and-update. Use `datetime.now()` (survives restarts, not `time.monotonic()` which resets).
- Evidence: CONV-20260227-0001 through -0021 (20+ identical alerts logged in production).

**4C. Static confidence disconnect** — Signal adapters set static confidence values that diverge wildly from learned Thompson reality. Example: `from_congressional_trade` sets confidence=0.75, while Thompson shows mean=0.164. The geometric mean formula applies Thompson weight to the adapter's static confidence — so 0.75 × 0.664 weight = 0.498, still misleadingly high for a source with 16.4% learned accuracy.
- Fix: Update `learning_config.py source_reliability` defaults to match Thompson means. `congressional: 0.20`, deprecate `sec_edgar` in favor of `sec_form4`.

---

### Priority 5: Session Sweep Bypass + Domain Independence + Persistence Hardening
**Consensus: 2/3 on bypass (Lead+Alpha for, Beta accepts with constraint), 3/3 on the rest**

**5A. Session sweep direct-output path** — MIDGE's best-documented edge (PF 1.84 from backtest, quality >= 0.65 Elite tier) is blocked by min_domains=3 because session sweeps fire in a single domain (technical). The convergence alerter requires two other unrelated domains to co-fire within 72 hours.
- Fix: Parallel direct-output path for signals with: DSR > 0.5 from completed backtest, >= 100 trades, quality >= 0.65. Currently only session_sweep_ifvg qualifies.
- Constraint: New sources require 30-day real-market validation to access the bypass (written as code comment, auditable).

**Note:** Alpha's concern about three sentiment sources gaming domain count was **disproven by code review** — `google_trends`, `stocktwits_sentiment`, and `social_sentiment` all emit `domain="sentiment"` and are already bucketed as one domain slot. The existing design handles this correctly.

**5B. Per-domain convergence windows** — Slow signals (COT weekly data, congressional 30-45 day lag) wash out of the global 72-hour convergence window before they can contribute. Add domain-specific windows: `positioning: 14 days`, `government: 7 days`, `contracts: 7 days`. Fast signals (technical, news) keep the 72-hour default.

**5C. Meta-learner cold-start bias** — `_seed_retirement_window_from_registry()` populates the retirement window with historical state, causing Wire 2 to tighten min_correlation based on historical retirements rather than live session performance. Fix: Mark seeded entries, ignore them in Wire 2 until 10 live-session entries accumulate.

**5D. Structural persistence hardening:**
- Hypothesis registry compaction (snapshot + incremental replay, prevents O(n) startup growth)
- OutcomeCollector `_registered` set age-prune (remove entries > 90 days old)
- Consistent atomic writes across all persistence files

---

## Additional Findings (Not Prioritized but Documented)

| Finding | Source | Severity | Notes |
|---------|--------|----------|-------|
| Thompson weight compression: 1.54x ratio between worst and best sources | Alpha | Medium | Weight formula `0.5 + dist.mean` maps [0,1] to [0.5,1.5]. Consider using `dist.mean` directly for mature distributions. |
| Rotation dilution: 19 sources / 3 ThreadPool slots / 50-step cadence = ~315 steps between any source's updates | Lead | Medium | Some sources go 5+ hours between fetches |
| Sell-side directional skew in Form 4 predictions: nearly all sec_form4 predictions are bearish | Lead | Low | Correct given mega-cap RSU selling; limits bullish convergence |
| Two-schema coexistence in predictions.jsonl: old format (entry_price=0.0) and new format coexist | Lead | Low | Old records can't be price-evaluated |
| Circadian rhythm operates in step-time, not wall-clock time | Alpha | Medium | ACTIVE phase fires during market closes with equal probability; requires test_mode override |
| TOCTOU race in hypothesis registry promote/retire | Beta | Medium | Background validation + agent-triggered validation without lock; will materialize as promotions increase |
| Organism tax: ~50 of 80 systems have no pathway to trading decisions | Alpha | Unknown | Requires StepTimer profiling before action; some serve as operational monitors |
| Endocrine signal dead-end: convergence alerts → dopamine → exploration bias → no trade | Alpha | Medium | Financial signals should exit through structured output, not hormone cascade |
| JSONL files grow unboundedly with no rotation | Beta | Medium | predictions.jsonl, outcomes.jsonl, hypotheses.jsonl — all append-only with no compaction |
| run_service.bat has no max restart count, no log rotation | Beta | Low | Infinite crash loop on environment failure |

---

## What the Triadic Process Caught

| Finding | Phase Found | What Solo Would Have Missed |
|---------|------------|----------------------------|
| finra_short is positive edge, not anti-signal | Phase 2→3 (Beta's base rate question → all revise) | Solo would have excluded MIDGE's strongest signal |
| Domain independence already implemented correctly | Phase 3 (Lead verifies code) | Solo would have built an unnecessary domain audit |
| Dedup race is fast-loop burst, not threading | Phase 2 (Lead + Beta combine evidence) | Solo would have fixed the wrong root cause |
| Reward misalignment teaches wrong objective | Phase 2 (Alpha + Lead converge) | Solo would have missed the VDN Q-table training implication |
| CF-1 is the same bug that caused prior corruption | Phase 1 (Beta) → Phase 2 (all agree) | Solo might have deprioritized a "JSONL append lock" |
| Static confidence × Thompson weight formula path | Phase 1 (Lead) → unchallenged | Only Lead traced the full geometric mean formula interaction |

---

## Dissenting Notes

### Beta's Sequencing Dissent (Minor)
Beta maintains that Priority 2 (paper trading) should be sequenced strictly AFTER Priority 4B (alert deduplication), not concurrently. Lead's compromise (build dedup gate INTO the paper trading path) is acceptable but creates two dedup mechanisms — one in the alerter and one in the paper trading book. Beta would prefer a single dedup layer in the alerter that all downstream consumers inherit. **This is a minor implementation preference, not a strategic disagreement.**

### Alpha's Thompson Weight Compression (Unresolved)
Alpha maintains that the Thompson weight formula `0.5 + dist.mean` compresses the quality spread too much (1.54x ratio between worst and best). Neither Lead nor Beta directly challenged this, but neither prioritized it. Alpha believes this should be Priority 3, not an additional finding. **The triad documented it but did not reach consensus on priority ranking.**

---

*This deliverable represents the collaborative output of three independent analysts across five phases of triadic work. All three agents endorse this document as representing the triad's collective findings.*
