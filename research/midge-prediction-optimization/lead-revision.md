# Lead Revision — Post Cross-Review

**Decision: REVISE**

After reading Alpha and Beta cross-reviews, I'm revising my priority ordering and specific recommendations. Here's what changed and why.

## Revised Priority Ordering

**Original #1:** Per-ticker convergence
**Revised #1:** Fix outcome duplication + deduplication guard (Alpha's evidence is compelling — corrupted learning compounds over time even if current impact is small)

**Original #2:** Fix congressional noise (min $15K)
**Revised #2:** Fix 10b5-1/RSU contamination + congressional $50K filter (accepting Beta's higher threshold)

**Original #3:** Wire VelocityDetector
**Revised #3:** Per-ticker convergence (still critical, now properly sequenced after data integrity fixes)

**Why the shift:** Alpha convinced me that building on corrupted foundations is worse than building on incomplete foundations. The outcome duplication bug, while small in current impact, establishes a pattern — every future scan would compound the error. Fix the data integrity layer first, then unlock the architecture.

## Revised Recommendations

### Congressional threshold: $15K -> $50K
Beta's transaction cost analysis is more rigorous than my heuristic. At 0.10-0.30% round-trip costs, a $15K trade signal needs to predict a move that exceeds the friction. $50K is the right minimum.

### Options flow: Still recommend, but AFTER multi-timeframe
Beta correctly identifies that options flow alpha decays in minutes-to-hours, which is incompatible with MIDGE's current daily scan cadence. Adding options flow before multi-timeframe architecture would waste the signal's time-sensitive information. Revised sequence: multi-timeframe first, then options flow into Tier 1.

### Confidence values: Accepting Alpha's critique
My original proposal (Thompson-weighted averaging) assumed Thompson distributions had useful posteriors. They don't. The right path is: (1) close outcome loop, (2) accumulate 50+ outcomes, (3) THEN weight by learned reliability. Until then, acknowledge that confidence values are informed priors, not calibrated probabilities.

### Contract symbol resolution: Adding to priorities
Alpha's finding that contract_award signals have symbol="" — breaking the feedback loop for that entire signal class — was a gap in my analysis. This needs to be fixed alongside the outcome collector, not as an afterthought. Without ticker resolution for contracts, the outcome collector will silently ignore 20% of the signal types.

### Regulatory compliance flag: Adding
Alpha's regulatory analysis for the prospective committee-to-award detector (my recommendation 4.2) deserves a compliance flag. Not blocking implementation, but any trade signal generated from congressional committee correlation should carry a visible warning.

## What I Stand Firm On

1. **Per-ticker convergence is still architecturally critical.** Alpha and Beta both confirmed it independently (different examples — RTX vs GD — but same conclusion). Moving it to #3 doesn't diminish its importance; it just sequences it after data integrity.

2. **Data source expansion matters.** Alpha frames my data source recommendations as "build more before fixing what's broken." That's a mischaracterization. I'm arguing that the system is missing its strongest potential convergence combinations (options + insider, dark pool + insider + congressional). But I accept the sequencing: fix first, expand second.

3. **8-K sentiment via Ollama remains high value.** Neither Alpha nor Beta engaged with this recommendation. The current rule-based item code mapping produces "neutral" for nearly everything. NLP-based sentiment on 8-K text would transform the events domain from noise to signal.
