# Devil's Advocate Findings: Three-Analyst Council for MIDGE
**Date:** 2026-03-13
**Analyst:** Devil's Advocate
**Proposal:** Replace single DeepAnalyst with three communicating analysts sharing a situation board

---

## Executive Summary

**The proposal is not harmful, but it risks solving the wrong problem.** The codebase analyst's findings are technically sound. The architecture is additive and safe. But four of the seven key challenges in the brief point to real failures that three analysts will not fix — and one failure mode is so severe it invalidates the entire proposal's value premise.

The strongest case against this proposal: **DeepAnalyst's output is already orphaned.** `ctx.inevitabilities` is set but confirmed to have no live consumer. Three analysts reading `ctx.inevitabilities` will produce three layers of analysis on top of data that goes nowhere. The situation board will be a second orphan. The human-readable output problem is not an analyst problem — it is a consumption and delivery problem.

---

## Assumption Audit

| Assumption | Status | Evidence |
|-----------|--------|---------|
| "DeepAnalyst runs every 200 steps" | VERIFIED | `market_hooks_steps.py` line 426, `if step % 200 == 0` block confirmed |
| "28 intelligence systems funnel through one analyst" | PARTIALLY CONTRADICTED | DeepAnalyst is one synthesis layer; it does not funnel all 28 systems. It reads the SIGNAL ARCHIVE, not the live intelligence pipeline. 9 of 28 systems (WorldModel, CascadeTracker, PatternLibrary, HypothesisEngine, etc.) have no confirmed connection to DeepAnalyst's output either |
| "The tiered alerters failed because output went nowhere" | CONTRADICTED | The actual failure mode is that tiered alerters receive NO SIGNALS. They check convergence on empty buffers. The output routing is correctly wired; the input is starved |
| "Three analysts add temporal awareness" | UNVERIFIED | The proposal assumes temporal awareness requires new analyst classes. The existing lag_correlations.json (69 entries) and Granger findings (2 entries) are already computed every 200 steps by existing systems. No analyst class is needed to surface them |
| "Human-readable output will improve" | UNVERIFIED | 32,665 plain-language alerts are already being written to `data/midge/alerts_human.jsonl`. The bottleneck is not output generation — it is that no human reads this file regularly |
| "Analysts communicating builds composite pictures" | UNVERIFIED | No mechanism is proposed for analyst-to-analyst feedback. The situation board is a write-once, read-later pattern, not a dialogue |
| "The proposal is performance-safe" | VERIFIED (conditional) | IF analysts read only pre-computed ctx data (not archives), the performance cost is negligible. This is the codebase analyst's recommendation but is not enforced by the proposal |
| "Template win rates are real signal" | CONTRADICTED | 14 DeepAnalyst runs in `inevitabilities.jsonl`. Win rates are 1.0 across AMD, NVDA, NOC, RTX — too early, too few graded outcomes. All template_win_rate values are either null or 1.0 from tiny samples. This scoring component produces noise, not signal |
| "The daemon is running at 1/3 target speed" | SUPERSEDED BY DATA | The daemon is not running at 0.18 steps/sec. In steady state (steps 43-50), it runs at ~1 step per 7 seconds = 0.14 steps/sec WITHOUT DeepAnalyst ever having run. DeepAnalyst was not invoked once in 50 steps because it requires step % 200 == 0. The current speed degradation is entirely caused by other factors |
| "Post-mortem insights are rich" | CONTRADICTED | `post_mortem_insights.json` contains only 4 combo_stats entries, 0 flagged orderings, 23 total outcomes with 43% "late" timing. This data is too sparse to support a TemporalPatternAnalyst with meaningful findings |

---

## Failure Modes

### Failure Mode 1: The Orphan Problem Compounds (Probability: HIGH)

**Evidence:** `ctx.inevitabilities` is confirmed as unread by any consumer (grep across full codebase, confirmed by codebase analyst). `market.intel.deep_analysis` EventBus channel has no confirmed subscriber. 150 inevitability entries exist in `data/midge/inevitabilities.jsonl` and 14 analyst runs have occurred — none of this data has informed a single trade or decision.

**If three analysts read this orphaned data, they produce three layers of analysis on top of data that changes nothing.** A `SituationBoard.get_snapshot()` written to `data/midge/situation.json` is a third orphaned file unless something actually reads it and acts on the findings.

**Root cause the proposal does not address:** The missing step is connecting analysis to action. DeepAnalyst's output isn't inert because one analyst isn't enough — it's inert because there's no wiring from `ctx.inevitabilities` to any decision gate, trade filter, or convergence weighting. Three analysts and a situation board also have no such wiring.

**Risk level:** The proposal builds a more complex version of the tiered alerter failure: sophisticated systems that produce output nobody consumes.

---

### Failure Mode 2: The Data Poverty Problem (Probability: MEDIUM-HIGH)

**Evidence:**
- Granger causality: 2 findings (finra_short → fred_macro, sec_efts → yfinance_price)
- Post-mortem combo stats: 4 entries, only 1 with a win rate below 1.0
- Template win rates: null or 1.0 from n<5 samples
- Lag correlations: 69 entries, dominated by finra_short/fred_macro pairs

**The TemporalPatternAnalyst's proposed inputs are nearly empty.** Its job is to synthesize `post_mortem_insights.json` + lag correlations + cascade energy ratios. With 4 combo stats and 2 Granger findings, it will produce a report that says "not enough data" on every run for the foreseeable future. This is not a flaw in the implementation — it is a data maturity problem. MIDGE needs months of live trading data before these inputs are rich enough to support a specialist analyst.

**The CausalChainAnalyst's proposed inputs are also thin.** WorldModel chains are curated by hand (114 nodes, 102 edges). The analyst would traverse the same static graph every 200 steps and produce identical output unless new edges are discovered. CascadeTracker has confirmed 0 cascade chains (based on daemon log — no CH_CASCADE_CONFIRMED messages seen).

---

### Failure Mode 3: The Performance Misdiagnosis (Probability: MEDIUM)

**Evidence from daemon log:**
- Steps 1-50: 952 seconds = 19 sec/step average
- Steps 43-50: 6-7 seconds per step at steady state (well above 2-sec pace)
- Step 41→42: 47 seconds (large API response completing)
- DeepAnalyst: 0 invocations in 50 steps (not yet reached step 200)

**The current performance bottleneck is not DeepAnalyst — it has never run in this session.** The 19-second average step time is caused by API fetch threads completing during the main loop (SQLite thread errors for raw_store confirm active writes from worker threads). The daemon is running at approximately 0.14 steps/sec (1/7 of a 2-sec pace), but this is caused by existing infrastructure, not DeepAnalyst.

**The three-analyst proposal is performatively safe only if analysts read pre-computed ctx data.** But the codebase analyst's recommendation is advisory, not enforced. If an implementer follows the brief's language about "reading all data sources," an analyst that opens JSONL archives will immediately double or triple the 200-step processing time.

**Unknown:** What does DeepAnalyst itself cost when it runs on 733,000 signals across 30 days? The brief says 0.18 steps/sec but this figure predates the log. The archive has grown from ~300K to 733K signals in the 30-day window. Loading 314MB of JSONL into memory at step 200 is an unmeasured risk.

---

### Failure Mode 4: The Signal Bias Doesn't Improve (Probability: HIGH)

**Evidence:** On 2026-03-13, the signal breakdown was:
- sec_form4: 38.6% (insider filings, not TA)
- ta_structure: 14.3%
- ta_bollinger: 9.7%
- sec_form8k: 7.3%
- openinsider_purchase: 7.1%
- ta_rsi: 4.5%, ta_candle: 4.4%

**Combined TA signals (ta_*): approximately 34%.** This contradicts the brief's "72% technical analysis" figure. The actual bias is toward SEC filings and insider data.

However, the more important finding: **specializing analysts by domain does not reduce this imbalance.** A CausalChainAnalyst that reads `ctx.inevitabilities` will see the same signal distribution that DeepAnalyst saw, because the inevitabilities were scored from the same archive. The domain balance of the archive is what it is — specialist analysts cannot reweight it by looking at its outputs differently.

---

### Failure Mode 5: The Template Win Rate Illusion (Probability: MEDIUM)

**Evidence:** In `inevitabilities.jsonl`, template_win_rate is either:
- `null` (template found but 0 outcomes)
- `1.0` (AMD, NVDA, NOC, RTX, RCL) from tiny samples (n≤2 likely)

In `_combo_boost()`, a template with win_rate=1.0 gets multiplier = `max(0.8, min(1.25, 0.8 + 0.9 * 1.0)) = 1.25`. This means the highest-ranked inevitabilities are boosted 25% by template win rates that are based on 1-3 outcomes. Any analyst that reads `ctx.inevitabilities` and treats `template_win_rate` as signal is consuming statistically invalid data.

**The ConvergenceQualityAnalyst's job is specifically to analyze the quality of DeepAnalyst's confidence scores.** If it does this correctly, it will immediately flag that template win rates are meaningless at current data volume and downgrade the scores of all top inevitabilities. This is useful — but it should have been a check inside DeepAnalyst, not a separate analyst.

---

### Failure Mode 6: Inter-Analyst Communication Is Not Designed (Probability: MEDIUM)

**What the brief says:** "Analysts should share findings and build composite pictures."
**What the proposal implements:** A situation board that analysts write to sequentially and a heartbeat writer reads for monitoring.

This is not communication — it is logging. If `CausalChainAnalyst` fires first and finds that NOC has strong downstream WorldModel links, there is no mechanism for `TemporalPatternAnalyst` to (a) know that NOC was flagged, and (b) prioritize searching NOC's timing data. The analysts run sequentially in `_run_analyst_council()` and the only shared state is the `SituationBoard` dict they write to.

**A situation board that is write-once per run is not a collaborative architecture.** It is three independent analysts and a shared log file. The inter-analyst communication described in the brief requires either (1) running analysts in multiple passes with feedback between passes, or (2) a shared working memory that analysts read mid-run. Neither is proposed.

---

### Failure Mode 7: The Anti-Pattern Risk — Building Rather Than Fixing (Probability: MEDIUM)

**Evidence from MEMORY.md:** "Stop building, start running — daemon must be alive 24/7, no new features until MIDGE proves herself."

**This directive was issued before this proposal.** The three-analyst architecture is a new feature. Three new analyst classes, a situation board, two bootstrap files modified, new channels, new tests. This is a medium-sized build project (likely 3-4 sessions, 1500-2000 lines of new code).

**What is broken right now that this doesn't fix:**
1. SQLite thread errors in raw_store (every SEC form4 fetch fails to store: confirmed in daemon log)
2. FRED macro directionality (2,361 signals all neutral — flagged in HANDOFF.md as unfixed)
3. API rate-limit backoff missing
4. 2 pre-existing test failures
5. `ctx.inevitabilities` is computed but consumed by nothing

The first four are operational failures that reduce data quality. The fifth is why the three analysts would produce orphaned output. None of these require building new analyst classes.

---

## Counter-Evidence Against the Proposal

### Counter-Evidence 1: The Simpler Intervention Has Higher Value

Adding one subscriber to `market.intel.deep_analysis` EventBus channel that reads `ctx.inevitabilities` and pushes the top 3 findings to the Alpaca/paper trading gate would immediately activate DeepAnalyst's output. This is approximately 30 lines of code. No new classes, no situation board, no new files.

By contrast, the three-analyst proposal adds 1,500+ lines, 5+ new files, and 2 modified bootstrap files to produce output that goes to `data/midge/situation.json` — which is again not wired to any decision system.

### Counter-Evidence 2: DeepAnalyst Already Does What Three Analysts Would Do

Reading `deep_analyst.py`: it already runs Thompson scoring (Analyst 1's job), template matching (pattern specialist), WorldModel chain traversal (CausalChainAnalyst's exact job), lag-lead scoring, density scoring, and historical win-rate lookup — all in one synthesized `Inevitability` object. The proposal is to split these six components into three specialist readers, then re-read the synthesized output. This is decomposition without recombination.

**The codebase analyst correctly identifies this as "reading pre-computed outputs."** But if the analysts only read DeepAnalyst's output and don't re-run any components, they add commentary on top of existing analysis — not new analysis. Commentary with no downstream consumer is documentation, not intelligence.

### Counter-Evidence 3: The Tiered Alerter Failure Is the Proof Case

Three tiered alerters were built. They were wired into the step cadence. They were given check_convergence calls every 10 steps. They write to `ctx._market_advisory`. And they have been running dead for months because they never received signal inputs.

**The failure pattern was not bad architecture — it was missing input wiring.** The three-analyst proposal carries the same risk: new analysts that receive the wrong inputs (or inputs too sparse to produce meaningful findings) will run silently and write empty reports.

---

## Alternative Approaches With Lower Risk

### Alternative 1: Wire DeepAnalyst's Output to a Decision Gate (3-5 hours, 50 lines)

Subscribe to `market.intel.deep_analysis`. When `ctx.inevitabilities` updates, take the top 3 with score > 0.70 and inject them as synthetic high-confidence convergence signals into the convergence engine. This closes the loop: DeepAnalyst's synthesis feeds back into the live pipeline. Zero new classes, zero new files.

**Risk:** Adds synthetic signals from historical analysis into the live convergence engine — could inflate confidence scores if not properly weighted. Score as a separate domain ("synthetic_inevitability") at 0.5 strength to avoid dominance.

### Alternative 2: Add Domain-Aware Scoring to DeepAnalyst (1 session, 150 lines)

Deepen DeepAnalyst with time-horizon specialization internally. Add a `horizon` parameter: `short_term` (1-5 days, weights density and momentum), `medium_term` (5-20 days, weights template and lag), `long_term` (20+ days, weights WorldModel and combo stats). Return three ranked lists from one `analyze()` call. Output goes to three sections of a single human report.

**This achieves the stated goal — three perspectives, temporal awareness, composite picture — without three separate classes, a situation board, or new bootstrap complexity.**

### Alternative 3: Fix What's Broken Before Building What's New

In priority order:
1. Fix SQLite thread safety in raw_store (WAL mode is set but connection objects are being passed across threads) — this is causing every SEC form4 batch to fail storage
2. Fix FRED macro directionality — 2,361 signals currently neutral
3. Wire inevitabilities to paper trading gate (30 lines)
4. Only then consider whether three analysts add value

**If MIDGE cannot store SEC data and cannot emit directional macro signals, the archive that three analysts would read is corrupted at the source.**

---

## What I Could Not Disprove

1. **The fractal law argument.** Law 2 (no bare dyads, minimum triads) and Law 7 (Rule of 3/5) do argue for three analysts over one. The architectural alignment with MIDGE's own laws is genuine. This cannot be dismissed.

2. **The specialization argument.** A dedicated `TemporalPatternAnalyst` that focuses exclusively on timing patterns across all post-mortem data would, over time (months), develop deeper timing intuition than a generalist DeepAnalyst that treats timing as one of six equal scoring components. The argument is sound — but it is a 6-month payoff on 1-week of build work.

3. **The situation board clarity argument.** Replacing `ctx._market_advisory` (an ad hoc dict with inconsistent keys) with a typed `SituationBoard` class would be a genuine architectural improvement. This is separable from the three-analyst question.

4. **The codebase analyst's blueprint is technically sound.** Reading only pre-computed ctx data, using try/except bootstrap, extracting to `market_analysts.py` — all of this is correct. The architecture does not violate any existing constraints.

---

## Gaps the Proposal Does Not Address

**Gap 1: Output consumption.** Who reads `data/midge/situation.json`? How does it reach Guiding Light? The proposal does not answer this. Without a delivery mechanism (push notification, dashboard, email, Alpaca annotation), the situation board is another unread file.

**Gap 2: Alert fatigue.** 32,665 human-readable alerts already exist in `alerts_human.jsonl`. Adding a situation board summary does not reduce the signal-to-noise ratio — it adds another layer. Three analysts × 200 steps × 24/7 = another 1,440+ reports per day at current pace.

**Gap 3: The SQLite thread error.** Every SEC form4 batch (10-90 trades per fetch) is failing to store in raw_store due to a thread-safety violation: "SQLite objects created in thread id 31640, used in thread id 30808." This means the raw store is not receiving most insider trading data. An analyst that reads this raw store for richer insider signals is reading incomplete data.

**Gap 4: Data maturity timing.** The proposal is architecturally ready. The data is not. Post-mortem insights will not be analytically meaningful until MIDGE has graded 100+ outcomes per combo type. At the current grading rate (23 total outcomes since launch), this is 3-6 months away. A TemporalPatternAnalyst built today will spend months producing "insufficient data" reports.

---

## Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Failure Probability** (10=unlikely) | 5/10 | Likely to be built correctly but unlikely to change outcomes because output consumption is not wired |
| **Failure Severity** (10=trivial) | 8/10 | Failure here is not catastrophic — it is additive waste. No existing system is harmed |
| **Assumption Fragility** (10=all verified) | 4/10 | Three key assumptions are contradicted: tiered alerter failure mode, signal bias source, data maturity |
| **Hidden Complexity** (10=none) | 5/10 | The SQLite thread error is an existing hidden problem that would affect any analyst reading raw_store. The archive growth to 733K signals in the 30-day window is unmeasured performance risk |
| **Overall Risk** (10=very safe) | 6/10 | Safe to build, but likely to deliver less value than estimated and may delay fixing actual operational failures |
| **Reversibility** (10=trivial) | 9/10 | Each analyst is behind a try/except. Removal is one delete and two line changes. Genuinely reversible |
| **Evidence Confidence** (10=rock solid) | 8/10 | All findings above reference specific files, line numbers, and live data. The unmeasured cost of DeepAnalyst on 733K signals is a genuine unknown |

---

## Red Flags

1. **`ctx.inevitabilities` has never triggered a single trade.** In 14 analyst runs across 2+ sessions, zero paper trades were registered from DeepAnalyst output. The three-analyst proposal assumes this output is valuable and needs better organization. It may need better routing instead.

2. **Post-mortem insights have 4 combo_stats and 0 flagged orderings.** The TemporalPatternAnalyst would be built on near-empty data. This is not a future risk — it is the current state.

3. **Template win rates are 1.0 everywhere.** This is a statistical artifact of low outcome volume, not a real signal. Any analyst that reads these values as signal quality indicators is amplifying noise.

4. **Granger causality has 2 findings.** Both involve the same source (finra_short) as the cause. The Granger analyzer needs more time and more diverse signal history to produce meaningful causal findings.

5. **The daemon is running at 0.14 steps/sec** without DeepAnalyst having run once. Adding three analysts at step 200 will create the first unmeasured performance event in this session.

---

## Bottom Line

**Build the situation board. Do not build three analysts yet.**

The situation board (`SituationBoard` class replacing `ctx._market_advisory`) is the only part of this proposal that addresses a real architectural debt — the ad hoc dict with inconsistent keys. It is small, clean, and genuinely useful for monitoring.

The three analyst classes are premature specialization. MIDGE does not have enough graded outcomes, enough Granger findings, or enough cascade data to make specialist analysts produce different output than a generalist would. They will run, they will have minimal data, and they will produce reports that say versions of "watching and waiting."

**The highest-value intervention is wiring `ctx.inevitabilities` to a decision gate — not building three analysts to read it more carefully.**

If the three analysts are built anyway, the codebase analyst's blueprint is the correct implementation path. The constraint is not architecture — it is timing.
