# Beta Revision — Post Cross-Review

**Decision: REVISE priority ordering and Kelly recommendation**

## Revisions

### Priority resequencing: Integrity before profitability
My original priority ordering was: (1) economic noise elimination, (2) 10b5-1 detection, (3) multi-timeframe, (4) decay rates.

**Revised ordering:** (1) Data integrity fixes (outcome dedup + contract symbol resolution), (2) Signal quality fixes (10b5-1 + congressional $50K filter), (3) Per-ticker convergence, (4) Multi-timeframe architecture, (5) Decay rate corrections, (6) Position sizing.

**Why:** Alpha convinced me that profitability metrics built on corrupted data are meaningless. Lead convinced me that per-ticker convergence unlocks the architecture's potential before multi-timeframe adds complexity. The resequencing layers: integrity -> signal quality -> architecture -> timing -> execution.

### Kelly criterion: Downgrading to simple rules
Alpha's cross-review argument is compelling: Kelly requires calibrated p (probability) and b (return ratio). MIDGE has neither. Applying Kelly to fabricated confidence values produces mathematically elegant but practically meaningless position sizes.

**Revised position sizing:** Simple rules-based framework:
- Base allocation: 5% of portfolio per convergence alert
- Scale by domain count: 2 domains = 0.5x, 3 domains = 1.0x, 4+ domains = 1.5x
- Hard cap: 15% per position
- Upgrade to Kelly after 100+ calibrated outcomes provide real hit rates

This is less elegant but more honest about what the confidence values actually mean.

### Decay rates: Acknowledging they're priors
Alpha's cross-review correctly notes my decay rate numbers are derived from studies on 1990s-2000s data. Market microstructure has changed significantly. I accept that my calibration table should be treated as informed priors for the Thompson Sampler, not as fixed constants. The two-component model (fast + slow decay) is structurally correct; the specific coefficients should be updated as outcome data accumulates.

### Contract symbol resolution: Adding to priorities
Alpha's finding that contract_award signals have symbol="" was a genuine gap in my analysis. I wrote extensive decay rate analysis for contract signals without noticing they can't participate in the feedback loop. This is the kind of finding that justifies the triadic protocol — I spent substantial analytical effort on a signal class that has a blocking upstream bug.

## Standing Firm

### Multi-timeframe architecture is structurally necessary
Neither Lead nor Alpha proposed this architecture. Lead's lag-correlation and my multi-timeframe are complementary, not competing. The case for separate convergence windows by signal type is supported by the decay rate analysis: you cannot put a signal with a 3-day half-life (8-K material events) and a signal with an 87-day half-life (SAM.gov opportunities) in the same convergence window and get meaningful analysis.

### Transaction cost awareness must exist
Neither Lead nor Alpha addressed whether signals are economically actionable. A signal that predicts a 0.5% move on a stock with 0.3% round-trip costs produces a net 0.2% expected return — barely worth the execution risk. MIDGE needs minimum economic viability thresholds per signal type.

### Leading vs. lagging classification is essential
My framework for categorizing signals as leading (actionable) vs. lagging (confirmatory) was not addressed by either analyst but is critical for knowing WHICH signals to act on vs which to use for model validation. Post-announcement contract awards are lagging — useful for updating Thompson distributions but not for generating new positions.

### Domain status table bug is real
I found `avg_strength` vs `strength` field name mismatch causing all-zeros display. Neither Lead nor Alpha caught this. Every scan report generated so far has had a broken domain status table. Small bug, but indicative of the gap between "looks like it works" and "actually works."
