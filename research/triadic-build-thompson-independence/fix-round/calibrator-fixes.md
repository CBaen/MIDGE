# Calibrator Fixes — Based on Independent Reviews

**Date:** 2026-03-06
**Reviews addressed:** review-1.md (Reviewer 1), review-2.md (Reviewer 2)

## No Fixes Required

Neither reviewer identified any critical or important issues in the Calibrator's code.

**Advisory findings noted (no action taken):**
- I/O latency in lock during forgetting log (R2) — performance concern for future, not correctness
- Cadence test uses simulation rather than importing actual hook (R2) — fragile but functional
- No multi-regime forgetting test (R2) — can be added incrementally
