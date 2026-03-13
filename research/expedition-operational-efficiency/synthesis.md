# Expedition Synthesis: MIDGE Operational Efficiency
## Date: 2026-03-12
## Vetted by: Orchestrator
## Alignment: Checked against Research Brief
## Validation: Phase 2 skipped — findings cross-corroborate across teams (Teams 1+4 confirm timing issue independently; Teams 1+3 confirm source name mismatch independently)

---

### High Confidence (teams converged with independent evidence)

**1. Thompson feedback loop writes to orphaned keys (Teams 1 + 4)**
- Sampling path uses `_ROTATION_TO_THOMPSON` to translate fetcher names → Thompson names (e.g., `ta_rsi` → `technical_rsi`)
- Feedback path does NOT apply this translation — `thompson_sampler.update("ta_rsi", success)` creates a new orphaned distribution
- The 146 seeded distributions that convergence actually reads NEVER receive updates
- **Evidence:** Team 1 traced the exact code path through `outcome_collector.py:180` → `outcome_tracker.py` → `thompson_sampler.update()`. Team 4 independently confirmed the forgetting gate blocks all decay because `current_evaluated` never changes.
- **Fix:** 3 lines in `outcome_tracker.py` — apply `_ROTATION_TO_THOMPSON.get(source, source)` before the update call

**2. Learning requires days; sessions last minutes (Teams 1 + 4)**
- Shortest outcome window: 5 days (sec_form8k). Most: 45-90 days
- At pace=2.0, 1 day = 43,200 steps. A 500-step session covers ~17 minutes of real time
- No grading can happen within any single daemon session
- Forgetting gate correctly protects Thompson from erosion during zero-learning periods
- **Evidence:** Both teams independently identified `OUTCOME_WINDOWS` in `outcome_collector.py:43-56` and the forgetting gate in `market_hooks_steps_core.py:187`
- **Fix:** Run `OutcomeCollector.evaluate()` at startup to grade predictions from prior sessions whose windows have elapsed

**3. Source name mismatches block both Thompson AND hypothesis validation (Teams 1 + 3)**
- Lag correlations reference `fred_macro`, `finra_short`, etc. Signal archive may use different source identifiers
- Hypothesis validator's `find_trigger_events()` finds zero matches if source names don't align
- Zero observations = permanent probation for hypotheses
- **Evidence:** Team 1 showed the `_ROTATION_TO_THOMPSON` map exists but isn't applied on feedback. Team 3 showed hypothesis validator at line 168-172 returns zero observations when source names don't match archive records

**4. Convergence engine is alive — 5,494 alerts, last at 05:07 today (Team 2)**
- The "zero alerts" report was a monitoring gap, not an engine failure
- Signal buffer contains 117,706 signals across 13 domains
- 11 bearish domains currently qualify above min_strength=0.6
- The 16.6-hour gap is consistent with daemon shutdown at 05:07, not engine failure
- **Evidence:** `data/market/alerter_state.json` shows `alert_counter: 5494`, last alert timestamps at 2026-03-12T05:07:34

**5. "Active" IS "promoted" — 4 hypotheses already promoted (Team 3)**
- The lifecycle is PROBATION → ACTIVE → HIBERNATED → RETIRED
- "Active" is the promoted state. There is no status beyond ACTIVE
- The `_hypotheses_promoted` counter resets each session — it's a session metric, not a lifetime metric
- **Evidence:** `hypothesis_registry.py:79-96` — `promote()` transitions PROBATION → ACTIVE. The 4 active hypotheses were promoted in prior sessions

---

### Prioritized Fix List

| # | Fix | Files | Lines | Impact | Confidence |
|---|-----|-------|-------|--------|------------|
| 1 | **Thompson namespace fix** — apply `_ROTATION_TO_THOMPSON` on feedback path | `outcome_tracker.py` | ~3 | Highest-leverage single fix. Unblocks ALL Thompson learning | Very High (4 teams) |
| 2 | **Startup outcome evaluation** — grade prior-session predictions at boot | `market_hooks.py` or bootstrap | ~10 | Closes cross-session learning gap. First grading within minutes of restart | Very High (2 teams) |
| 3 | **Activate auto causal stories** — fallback before "REQUIRES MANUAL REVIEW" | `hypothesis_causal.py` | ~5 | Unblocks 26 probation hypotheses | High (Team 3) |
| 4 | **Wire per-ticker convergence** — `convergence_ticker.py` into main sensing loop | `market_hooks_steps_core.py` or `sensing_collector.py` | ~15 | Surfaces ticker-specific inevitabilities (what Guiding Light wants) | High (Team 2) |
| 5 | **Map unknown domain sources** — add to `_SOURCE_DOMAIN_MAP` | `pattern_library.py` | ~5 | Recovers 751 misclassified bearish signals at avg strength 0.824 | Medium (Team 2) |
| 6 | **Fix FRED macro directionality** — emit directional signals for yield curves | `fred_client.py`, `fred_models.py` | ~20 | Recovers macro domain for convergence (2,361 signals currently neutral) | Medium (Team 2) |
| 7 | **Add API rate-limit backoff** — per-source failure tracking | `sensing_hook.py` | ~15 | Prevents domain starvation from rate-limited APIs | Medium (Team 4) |
| 8 | **Surface forgetting skip at INFO** — visibility into learning state | `market_hooks_steps_core.py:193` | 1 | Free observability improvement | Low effort (Team 4) |
| 9 | **Verify source name alignment** — cross-ref lag_correlations vs signal archive | diagnostic only | 0 | Confirms or eliminates secondary blocker for hypotheses | High (Teams 1+3) |

---

### Architectural Insight: Two Separate Intelligence Pipelines

**DeepAnalyst** = historical synthesis engine (30-day archive, per-ticker, multi-domain stacks)
**ConvergenceAlerter** = live pattern detector (72h rolling window, GLOBAL not per-ticker)

DeepAnalyst finds NOC/LMT/GD with 5-domain stacks. ConvergenceAlerter fires ONE global bearish alert picking whatever ticker happens to have metadata. Neither is broken — they operate on different data, different time horizons, different scopes.

**The highest-value architectural fix** is bridging these: when the signal buffer shows ticker X has 3+ independent domains converging, fire a ticker-specific alert. This is exactly what `convergence_ticker.py` was built for — it just isn't wired into the main loop.

---

### What's Actually Working

- Signal ingestion: 117,706 signals across 13 domains, 31 data sources, all fresh
- Convergence engine: 5,494 alerts fired historically, architecture sound
- Hypothesis generation: 30 generated, 4 promoted to active, lifecycle working
- Meta-learning gates: permissive, would loosen thresholds if needed
- Forgetting gate: correctly protects Thompson from erosion
- Signal persistence: buffer saves/loads across daemon restarts
- WitnessNotifier: 570K+ witnessed, 0 failures
- EventBus wiring: 150 channels, ~40 witnesses active

---

### What's Silently Failing

- Thompson learning: 146 distributions at prior, zero updates reaching seeded keys
- Cross-session feedback: predictions registered but never graded (outcome windows > session length)
- Hypothesis promotion: 26 blocked by causal story gate (categorical, not threshold)
- Per-ticker convergence: exists in code, not wired to main loop
- FRED macro: 2,361 signals, effectively all neutral (zero bullish above 0.6, one bearish)
- Unknown domain: 751 signals from unmapped sources inflating domain count
- API rate limiting: no backoff, failed fetches silently dropped

---

### Filtered Out

1. **"Revert growth sprint cadences"** — Research Brief explicitly forbids this. Teams correctly did not suggest it.
2. **"Remove the forgetting gate"** — It's working correctly. Without it, Thompson would erode to Beta(2,2) floor on every distribution.
3. **"Restructure circular architecture"** — Destructive boundary. The architecture is sound; the wiring has gaps.

---

### Risks

1. **Fix 1 (Thompson namespace)** — Low risk. The `_ROTATION_TO_THOMPSON` map already exists and works on the sampling side. Applying it to the feedback side is symmetric and safe.
2. **Fix 2 (Startup evaluation)** — Medium risk. Must ensure predictions from prior sessions are still valid and not double-graded. Need idempotency check.
3. **Fix 3 (Auto causal stories)** — Medium risk. Auto-generated stories face +0.01 win rate penalty (0.53 vs 0.52). May produce low-quality causal narratives that pass the gate but don't add insight.
4. **Fix 4 (Per-ticker convergence)** — Low risk. The code exists in `convergence_ticker.py`. Wiring it in is additive, not modifying existing behavior.
