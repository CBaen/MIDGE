# Alpha Revision — Post Cross-Review

**Decision: STAND FIRM on core thesis, REVISE specific recommendations**

## Standing Firm

### Confidence calibration is the deepest problem
Lead's cross-review acknowledges this but reframes it as "practical path forward is still close the outcome loop first." Beta's cross-review partially diverges, arguing the hardcoded values are "reasonable priors" not "fabricated."

I stand firm: the additive confidence formula (`0.5 + 0.1 * categories + 0.1 * strength`) is the fabrication, not the individual base values. Two 70% signals combined additively produce 0.90 confidence, but two independent 70% events have a 49% joint probability. The formula inflates confidence by design. This matters because position sizing (Beta's Kelly framework) will use these numbers. Garbage confidence -> garbage position sizes -> losses.

**Evidence reinforcing my position:** Beta's own cross-review admits Kelly criterion "requires calibrated probability estimates (p)" and proposes falling back to simple rules-based sizing because calibration doesn't exist yet. This CONFIRMS my central thesis — the confidence numbers can't support the mathematical framework built on them.

### Outcome duplication is Priority 1 for the learning system
Lead revised to accept this as #1. Beta acknowledges it but rates impact as "small right now." I maintain: the deduplication fix is 5 lines of code. The damage it prevents over 50+ future scans is cumulative. This is the cheapest fix with the highest long-term leverage.

## Revisions

### Regulatory risk: Scaling back
Beta's cross-review argues the SEC enforcement risk is overstated for retail individual use. After reflection, I agree the Advisers Act 204A concern is primarily institutional. The MNPI mosaic theory under Rule 10b-5 is more relevant but still low-probability for a retail system using exclusively public data.

**Revised position:** Add a compliance flag to committee-correlation signals (as Lead proposed in revision). Don't block the feature. The user should be informed, not prevented.

### Decay rates: Accepting Beta's framework
I didn't address decay rates in my Phase 1 analysis. Beta's calibration table is well-sourced from academic literature. I accept their two-component decay model (fast market reaction + slow fundamental drift) as a better model than single-rate exponential. My only reservation: the specific numbers (0.035 for insider, 0.025 for clusters) are derived from 2001 data and should be treated as priors, not constants.

### Multi-timeframe architecture: Adding to my recommendations
Beta's three-tier convergence architecture (Tactical 48h / Strategic 21d / Thematic 90d) is an important structural improvement I didn't propose. It directly addresses the timescale conflation problem that corrupts convergence quality. I'm adding this to my recommendations with one modification: cross-tier convergence should have a REDUCED confidence boost (not additive with within-tier convergence) because signals at different timeframes responding to the same event are not independent evidence.

### Position sizing: Accepting need, rejecting Kelly
I maintain that Kelly criterion is premature. But Beta is right that MIDGE needs to answer "how much?" The compromise: simple rules-based sizing now (5% base, 2x for 3+ domain convergence, 0.5x for single domain, hard cap at 15%), Kelly criterion after 100+ calibrated outcomes provide real p and b values.
