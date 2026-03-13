# Council Synthesis: Communicating Multi-Analyst Architecture
## Date: 2026-03-13
## Vetted by: Orchestrator
## Alignment: Checked against Research Brief

---

## The Recommendation (Sequenced, Not All-at-Once)

The council converged on a **three-phase approach** that respects both the architectural vision and data maturity reality:

### Phase A: Wire What Exists (immediate — ~50 lines)
1. Subscribe to `market.intel.deep_analysis` EventBus channel → feed top 3 inevitabilities into paper trading gate as Validator 3
2. Feed signals to the three tiered alerters (they're starved — 3 lines in sensing hook)
3. Fix the SQLite thread safety error in raw_store (SEC form4 data is failing every fetch)

### Phase B: Build the SituationBoard (next — ~200 lines, 1 new file)
1. Create `mae_core/market/intelligence/situation_board.py` — typed class replacing `ctx._market_advisory`
2. Thread-safe (RLock), structured slots, `publish(analyst_id, finding)` / `get_snapshot()` interface
3. Human-readable output via `data/midge/situation.json` (overwritten each cycle)
4. Wire existing systems to write to it: DeepAnalyst results, convergence alerts, cascade tracker state
5. This is valuable independent of whether analysts are built — it cleans up architectural debt

### Phase C: Build Three Analysts (when data matures — ~400 lines each, 3 new files)
**Gate:** Post-mortem has 50+ combo_stats AND Granger has 10+ findings AND 100+ graded convergence outcomes
1. **CausalChainAnalyst** — reads `ctx.inevitabilities` + WorldModel + CascadeTracker. Produces causal narrative.
2. **ConvergenceQualityAnalyst** — reads inevitabilities confidence distribution + domain status + post-mortem insights. Quality-checks the convergence engine's output.
3. **TemporalPatternAnalyst** — reads SituationBoard (other analysts' findings) + lag correlations + cascade energy ratios. Produces timing/energy meta-analysis. This is the novel contribution — no existing framework has this.

All three read pre-computed ctx data only. Zero archive I/O. Bootstrap in `market_analysts.py`. Run every 200 steps via `_run_analyst_council()`.

---

## Shared Dimensions (Cross-Agent Comparison)

| Dimension | Codebase Analyst | External Researcher | Devil's Advocate | Avg | Spread |
|-----------|-----------------|---------------------|------------------|-----|--------|
| Overall Risk (10=safe) | 7/10 | 7/10 | 6/10 | 6.7 | 1 |
| Reversibility (10=trivial) | 10/10 | 8/10 | 9/10 | 9.0 | 2 |
| Evidence Confidence | 9/10 | 8/10 | 8/10 | 8.3 | 1 |

**Interpretation:** Low spread across all shared dimensions. The council broadly agrees this is medium-risk, highly reversible, and well-evidenced. The small risk gap (6 vs 7) reflects timing disagreement, not architectural disagreement.

---

## High Confidence Findings (2+ agents converged independently)

1. **`ctx.inevitabilities` is orphaned** — confirmed by all three via independent grep. DeepAnalyst runs but nothing acts on its output. This is the highest-priority fix.

2. **Tiered alerters failed from signal starvation, not bad wiring** — confirmed by CA (code trace) and DA (daemon log). The fix is feeding them signals, not replacing them.

3. **SituationBoard replaces `ctx._market_advisory`** — all three recommend this as an independent architectural improvement. Thread-safe, typed, structured.

4. **Performance is safe if analysts read only pre-computed data** — all three agree. The constraint is that analysts must NOT re-read JSONL archives.

5. **Temporal Analyst is the novel contribution** — ER found no existing framework with a dedicated temporal/timing specialist. CA confirmed CascadeTracker energy ratios are unread. DA acknowledged this is correct long-term but premature now.

---

## Disagreements

### "Build now" vs "Build after data matures"
- **CA/ER position:** Build the architecture now so it's ready when data matures. Sequential analysts (Gen 1) can produce partial findings immediately.
- **DA position:** Building analysts on sparse data produces "insufficient data" reports for months. Fix operational failures first. The architecture can wait.
- **Synthesis:** Both are right. The ARCHITECTURE (SituationBoard) should be built now. The ANALYSTS should wait for data maturity gates. This is the sequenced approach.

### "Enhanced DeepAnalyst" vs "Three new analysts"
- **DA position:** Add domain-aware scoring (short/medium/long horizon) to DeepAnalyst. One class, three perspectives.
- **CA/ER position:** Specialization enables depth impossible in a generalist. Six equal-weight components prevent any one from going deep.
- **Synthesis:** The DA's alternative is viable as an interim step. But it violates Law 2 (triadic structure) and doesn't enable inter-analyst communication, which is the stated goal. The three-analyst approach is architecturally correct for MIDGE's laws.

---

## Filtered Out

1. **ER's proposal to make ConvergenceAlerter consume SituationReports instead of raw signals** — violates the brief's constraint (do not modify convergence engine). Flagged by DA. Correct long-term vision but requires a separate council decision.

2. **Counter-Analyst / Bear Desk** — interesting but adds a 4th analyst, violating Law 2 (triadic, not quaternary). Could be implemented as a mode flag on existing analysts rather than a new class.

3. **SRMT Global Broadcast Memory** — circular reasoning risk (A conditions on B who conditioned on A). Interesting for a future evolution, not a first build.

---

## Risks

1. **Output consumption (from DA):** If `data/midge/situation.json` is written but nobody reads it, we've built a second orphan. **Mitigation:** Phase A wires inevitabilities to the paper trading gate FIRST, proving the consumption path before building more output.

2. **Data maturity (from DA):** Post-mortem has 4 combo stats, Granger has 2 findings. **Mitigation:** Phase C is gated on data maturity thresholds, not a calendar date.

3. **SQLite thread safety (from DA, confirmed in daemon log):** SEC form4 data is failing to store every fetch due to cross-thread SQLite access. This corrupts the data that any analyst would read. **Mitigation:** Fix this as part of Phase A.

4. **Performance at 733K signals (unknown):** DeepAnalyst's cost when loading the full 30-day archive (now 733K signals, growing) has never been measured. **Mitigation:** Analysts read `ctx.inevitabilities` (pre-computed), not raw archives. But DeepAnalyst itself may need optimization.

---

## Research Files

All council research is preserved at:
```
research/council-analyst-architecture/
  research-brief.md
  codebase-analyst-findings.md
  external-researcher-findings.md
  devils-advocate-findings.md
  codebase-analyst-challenge.md
  external-researcher-challenge.md
  devils-advocate-challenge.md
  score-extraction.md
  synthesis.md
  tension-report.md (pending)
```
