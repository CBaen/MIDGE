# Witness Beta Revision: Phase 3 — Post-Cross-Review Position

**Role: Witness Beta — Adversarial Lens**
**Phase: 3 Revision**
**Date: 2026-02-22**
**Status: REVISE — with specific updates to four positions**

---

## Framing

After reading all three cross-reviews, I am revising four positions and standing firm on the rest. The cross-review process surfaced two types of updates: places where peer reasoning exposed a gap in my original analysis, and one place where I initially miscategorized a finding and now accept Lead's self-correction. I am also establishing a final priority order that synthesizes all three investigations into a single implementation sequence.

The most important meta-finding from the cross-review: the three investigations were genuinely orthogonal. Lead worked from data shapes outward. Alpha worked from bootstrap mechanics inward. I worked from execution behavior and failure paths. None of us could have produced the combined picture alone. The five new failure modes I found in my cross-review (Section 5 of my cross-review document) only become visible when you hold all three views simultaneously.

---

## Part 1: Revised Positions

### 1.1 ACCEPT — Lead's Self-Correction on ConvergenceAlerter Category-Counting

**What I found originally (Section 1.3 of beta-findings.md):** I flagged `min_domains=2` as dangerous because two signals from the same company in two correlated domains ("insider" and "government") can trigger a convergence alert. I also, in my initial read, had a concern about whether the `categories_seen` logic was correctly deduplicating by category. I marked the confidence boost formula as a potential double-counting risk.

**What Lead's self-correction says:** Lead accepted my `min_domains=3` recommendation but also pointed out (in their cross-review, Part 1.5) that the `categories_seen` logic — using `domain_categories` map — IS correctly deduplicating by category. Five "market" domain signals contribute only 1 to `cross_domain_count`. The formula is more careful than I initially gave it credit for.

**My verification:** I re-read `convergence_alerter.py` lines 239-248 directly:
```python
category = self.domain_categories.get(domain, domain)
categories_seen.add(category)
...
cross_domain_count = len(categories_seen)
```
Lead is correct. The domain_categories map collapses multiple market-type signals into a single "market" category. The `confidence_boost` calculation uses `cross_domain_count = len(categories_seen)`, not `len(domains_seen)`. Five signals from different market domains still contribute only 1 to the boost count.

**Updated position:** I accept Lead's self-correction on the boost formula. The category-counting is correctly implemented. My concern in the original findings about potential double-counting through the boost formula was wrong. The `min_domains=2` concern stands and my recommendation to raise it to 3 remains correct — but the reason is entity-level correlation (same company), not formula miscalculation.

**What this changes:** Section 1.3 of my original findings should read: "The category-based boost correctly collapses domain signals into category signals. The problem is exclusively in `min_domains=2` allowing same-entity correlated signals to trigger convergence — not in how the boost is calculated."

---

### 1.2 ACCEPT WITH EXTENSION — Alpha's Fractal Placement as K3 Repair

**What I found originally:** I did not address fractal placement at all in my original findings. In my cross-review, I noted that Alpha's biological coherence argument holds and that the concern about hollow K3 nodes from the housestockwatcher/usa_spending/sam_gov/correlation_tracker residuals is real but manageable.

**What Alpha's finding adds:** Alpha identified (Part 4 and Part 12) that `organ-cluster-cognitive` currently has only 2 children — `cognitive-system` and `sensory-system`. This is a pre-existing Law 1 violation. Adding `market-intelligence-system` as the third child does not just extend the hierarchy; it repairs a bare dyad. Lead confirmed this in their cross-review (Section 1.2, Part 4, Surprise 1) and endorsed it without qualification.

**Does this address any failure mode I found?** Partially, and in an unexpected way.

My original finding in Section 3.1 was that the hardcoded Qdrant URLs are architectural violations because the actual data flow bypasses ConnectionRegistry — the registered connections are not the paths the data travels. This remains true. But Alpha's fractal placement finding changes the context: the violation is not just an implementation shortcut — it is happening inside an organ that currently has a structural deficiency. Fixing the Qdrant routing and completing the K3 are related acts. The organism's cognitive cluster is currently a dyad that also has unwitnessed data flows running through it. Both need to be addressed together.

**What this changes:** I am adding to my position that the Qdrant localhost fix and the K3 completion must be implemented together as a single structural repair. Fixing only the Qdrant URLs while the cognitive cluster remains a dyad leaves a different Law 1 violation in place. Completing the K3 while leaving the Qdrant calls hardcoded means the new triad is formally correct but functionally hollow — the witnesses register as observers of paths that no data travels.

**Updated position:** Alpha's K3 insight is correct and addresses the structural context I was missing. It does not fix any of the specific failure modes I identified, but it changes the implementation ordering: the fractal repair (K3 completion) and the Qdrant routing fix must be treated as a single coupled change, not two separate items.

---

### 1.3 REVISE — The OutcomeTracker Gap: Most Critical Component?

**What I found in my cross-review (Section 5.3):** The feedback loop requires an OutcomeTracker — a component that (N days after a prediction is made) checks the outcome against `price_fetcher_for_outcomes()` and calls `ThompsonSampler.update()`. This component does not exist. Without it, predictions accumulate in `predictions.jsonl` with no evaluation, `ThompsonSampler.update()` is never called from real outcomes, and the Bayesian distributions are frozen at their seeded values. I assessed this as the single most dangerous gap in terms of missing design intent.

**What Lead's cross-review adds (Section "What Beta Missed"):** Lead pointed out that `price_fetcher_for_outcomes()` (line 263 in `price_fetcher.py`) already exists and is architecturally shaped as the Bayesian feedback function. It accepts a symbol and a prediction date and returns price change over a forward window. The feedback loop implementation already has one of its legs built. What is missing is not the price-checking function but the scheduling mechanism that calls it N days later and routes the result to the Thompson Sampler.

**Does this change whether OutcomeTracker is the most critical missing component?**

This requires careful reasoning. The missing OutcomeTracker means the learning loop never closes. The Thompson Sampler's Bayesian state can never be updated from real market outcomes. Everything else in the architecture — the edge detectors, the convergence alerter, the bootstrap wiring — produces outputs that are never validated against ground truth.

Alpha's FRL-to-Thompson feedback (Part 11 of alpha-findings) partially compensates: FRL rewards could trigger Thompson updates when agents succeed. But this is organism-level feedback ("did the agent achieve its goal?"), not signal-level feedback ("did this specific market signal predict correctly?"). These are different feedback loops and only one closes the market prediction loop.

However: Lead is right that I overstated the gap by treating the price-checking function as entirely absent. `price_fetcher_for_outcomes()` exists. The missing piece is the scheduler and the `predictions.jsonl` reader — a relatively small component given that its constituent functions already exist. This is meaningfully less work than building the whole loop from nothing.

**Updated position:** The OutcomeTracker gap is still the most critical architecturally-missing component for MIDGE's core value proposition (learning signal reliability from outcomes). But Lead's finding reduces the implementation cost estimate from "build a complete feedback loop" to "build the scheduler that connects an existing function to an existing database." This changes the priority ordering: the OutcomeTracker is now a Phase 1 item rather than deferred, because the two hardest parts (price fetching, Bayesian update) already exist. What is missing is the glue: a component that reads `predictions.jsonl`, identifies predictions that are N days old, calls `price_fetcher_for_outcomes()`, and routes success/failure to `ThompsonSampler.update()`. This could be as simple as 60-80 lines of Python.

**Implication for priority order:** OutcomeTracker moves up in the list (see Part 3 below).

---

### 1.4 PARTIALLY REVISE — The `prior_scale=1` Recommendation

**What I found originally (Section 1.1, beta-findings.md):** I identified that `DEFAULT_PRIOR_SCALE = 10` is dangerous: it seeds `sec_edgar` at Beta(9.5, 0.5) with variance 0.004, which is tighter than a distribution derived from 10 real observations. The system starts with fake confidence. I recommended `DEFAULT_PRIOR_SCALE = 2`.

**What Alpha adds (Divergence 3 of alpha-cross-review):** Alpha accepted my `prior_scale=2` recommendation and added a second required change I had not stated explicitly: `min_variance` in `get_uncertain_signals()` must be lowered from 0.01 to 0.001 simultaneously. Alpha's reasoning: changing prior_scale alone without changing the exploration threshold still produces the lock-in problem for the high-reliability seeds. Beta(1.9, 0.1) with `prior_scale=2` has variance `(1.9 * 0.1) / (4 * 3)` = 0.0158, which is above the current 0.01 threshold. But `prior_scale=1.5` would produce Beta(1.425, 0.075) with variance 0.0224 for the 0.95-reliability signal, which is comfortably above 0.01. So at `prior_scale=2`, the exploration threshold is actually not the binding constraint — the tight prior at `prior_scale=2` has variance ~0.016, which barely passes 0.01.

**My verification:** Computing directly for `prior_scale=2`, `sec_edgar` at reliability 0.95:
- alpha = 0.95 * 2 = 1.9
- beta = 0.05 * 2 = 0.1
- variance = (1.9 * 0.1) / ((2.0)^2 * 3.0) = 0.19 / 12.0 = 0.01583

This is above 0.01, so `min_variance=0.01` would NOT block exploration at `prior_scale=2`. Alpha's concern is correct in principle but the arithmetic shows `prior_scale=2` clears the threshold. The two changes need not be made together if `prior_scale=2` is chosen — but Alpha's point remains valid that they should be made together if anyone chooses a smaller prior_scale (e.g., `prior_scale=1`).

**Lead's cross-review adds (Section "What Lead Missed"):** Lead also accepted the `prior_scale` finding and confirmed the `min_variance` interaction.

**Was my original recommendation (`prior_scale=2`) correct?**

Yes. However, I want to refine the framing. My original recommendation stated the goal as "reducing prior overconfidence." The more precise framing is that `prior_scale` should be chosen such that the seeded distribution's variance exceeds the exploration threshold. For the highest-reliability signal (sec_edgar at 0.95):
- `prior_scale=1`: Beta(0.95, 0.05) — variance = (0.95 * 0.05) / (1.0 * 2.0) = 0.02375. Passes 0.01 threshold.
- `prior_scale=2`: Beta(1.9, 0.1) — variance = 0.01583. Passes 0.01 threshold, marginally.
- `prior_scale=3`: Beta(2.85, 0.15) — variance = 0.01018. Passes 0.01 threshold barely.
- `prior_scale=4`: Beta(3.8, 0.2) — variance = 0.0076. BLOCKED by 0.01 threshold.

This means: for `prior_scale <= 3` with `sec_edgar` at 0.95, the seeded distributions will be eligible for exploration. For `prior_scale >= 4`, they will be locked out. My recommendation of `prior_scale=2` is correct and provides a margin above the exploration threshold. `prior_scale=1` is more honest (claims only 1 synthetic observation) but the signals become so wide that the sampler effectively can't distinguish between reliable and unreliable sources until real data accumulates. `prior_scale=2` is the right balance.

**The companion change:** Alpha is correct that `min_variance=0.001` should be set as a floor, not because `prior_scale=2` requires it, but because future changes to `prior_scale` or `source_reliability` values might inadvertently create distributions that fall below the current 0.01 threshold without anyone noticing. Lowering the exploration threshold provides a safety margin against miscalibrated seeds in the future.

**Updated position:** `prior_scale=2` is the correct recommendation. Additionally, `min_variance` should be lowered to 0.001 as a defensive measure — not because `prior_scale=2` alone requires it, but to prevent the exploration lock-in pattern from recurring if prior_scale is ever tuned upward.

---

## Part 2: Positions I Am Holding Firm

### 2.1 EventBus Channel Name Mismatch Is the Top Implementation Risk

My cross-review (Section 5.1) found that Lead proposes `market.edge.cluster_detected` while Alpha proposes `market.cluster_signal`. These are different strings. The publish side and subscribe side will never connect without explicit reconciliation.

Lead did not address this in their cross-review. Alpha did not address it either. No one has proposed a canonical channel namespace.

**I am holding firm on this being Priority 0** — it must be resolved before any EventBus wiring code is written. It is not enough to have both a correct signal format (Lead's `MarketSignal`) and a correct bootstrap design (Alpha's Layer 33) if the channel names through which they communicate are mismatched.

**My recommendation remains:** Adopt a hierarchical namespace with two tiers:
- Tier 1 (edge output): `market.edge.{signal_type}` — e.g., `market.edge.cluster`, `market.edge.filing`, `market.edge.contract`
- Tier 2 (intelligence output): `market.intel.{output_type}` — e.g., `market.intel.velocity`, `market.intel.convergence`, `market.intel.alert`

Either Lead's tier naming or Alpha's flat naming could work, but one must be chosen before code is written.

---

### 2.2 ContractPredictor + ConvergenceAlerter Double-Counting Is a Design Flaw, Not an Implementation Bug

My cross-review (Section 5.4) found that ContractPredictor publishes a synthesized `ContractPrediction` that will be recorded by ConvergenceAlerter as a "contracts" domain signal. This ContractPrediction is derived from insider signals that ConvergenceAlerter also receives directly. The same insider buy appears twice: once as an "insider" domain signal, once baked into the "contracts" domain signal from ContractPredictor.

Lead's proposed mitigation (raising `min_domains` to 3) does not solve this — it only raises the bar for the double-counting to trigger.

**I am holding firm on this as a design-level problem.** The correct fix is architectural: ContractPredictor should either (a) publish its component inputs to EventBus and let ConvergenceAlerter synthesize them, OR (b) publish its confidence score to a distinct channel that ConvergenceAlerter treats as an already-synthesized signal (and therefore does NOT combine with the raw component signals). Option (b) is simpler to implement and preserves ContractPredictor's domain-specific formula. The correct implementation for option (b): ConvergenceAlerter should have a flag or separate domain ("contract_prediction") that maps to "institutional_synthesis" category — a category that does not overlap with "insider" or "contracts" category. This way the ContractPrediction is a unique input, not a recombination of existing inputs.

---

### 2.3 No Forgetting Mechanism Degrades Over Time, Not Gradually

My finding (Section 1.1, beta-findings.md) that ThompsonSampler has no time-based decay remains unaddressed by both Lead and Alpha. Lead's architecture assumes the Thompson Sampler is a reliable weighting authority. Alpha's FRL-to-Thompson feedback feeds successes but never decrements stale beliefs.

The failure mode is not immediate — it is a slow drift. A signal that worked in Q1 2025 accumulates high alpha. The market regime changes in Q3 2025. The signal now generates noise. But the Thompson Sampler still samples it at high probability because alpha never decays. The system locks onto a once-good signal that is now broken.

**I am holding firm that this is a Critical Integration Hazard**, not just a tuning issue. The fix is one method: `_apply_forgetting(decay_factor)` that multiplies alpha and beta by a factor < 1, called daily. This preserves the mean direction while shrinking total weight toward the prior. The decay rates in `learning_config.py` were designed for exactly this purpose but are currently dead code. Connecting them is straightforward.

---

### 2.4 The `discovery_log.jsonl` Is a Write-Only Diary

Lead identified this in their cross-review (Section "What Beta Missed"). Lead notes that ConvergenceAlerter is supposed to write novel pattern discoveries to `data/market/discovery_log.jsonl` but no code reads this file.

I did not catch this in my original investigation — I did not examine the file directly. Lead caught it. But I am adding it to my standing position: a write-only log that claims to be a "pattern library" is misleading documentation. CLAUDE.md describes it as "Learned Bayesian distributions and historical predictions." The discovery log cannot do its described job without a reader.

**I hold this as a documentation mismatch that needs either (a) a reader that uses discovery log patterns to seed future signal weighting, or (b) honest documentation that the log is audit-only.** It is not a crash bug but it is misleading architectural documentation that will confuse the next instance who reads it expecting a live pattern library.

---

### 2.5 The `jobs_30d / 30` Baseline Is a False Spike Factory

My original finding (Section 2.6): `jobs_30d` is derived from a 7-day API call (`date_posted: "week"`). The denominator of the spike detection formula (`daily_avg = signal.jobs_30d / 30`) is therefore 4x lower than a true 30-day average. The leading indicator chain that produces MIDGE's pre-announcement edge fires as a false positive almost universally.

Lead confirmed this in their cross-review (Section "Surprises") and noted it has a larger architectural consequence than I described: the SAM.gov → hiring blitz → contract award signal chain is MIDGE's primary edge. If hiring spikes are systematically false, the chain is broken at its most sensitive node.

**I stand firm on this as Priority 3** (after is_purchase crash bug and thompson_distributions rebuild). The fix is: use `daily_avg = signal.jobs_7d / 7`, not `signal.jobs_30d / 30`. This requires either a second API call to get true 30-day data, or an honest correction to a 7-day baseline. The 7-day baseline is less stable but honest. The 30-day baseline is more stable but currently impossible given the API only returns 7-day data.

---

## Part 3: Revised Priority Order

This is the complete implementation priority, synthesizing all three investigations and incorporating all Phase 3 revisions. Lead's Phase 3 revision should be consulted for the signal architecture sequence; Alpha's revision should be consulted for the bootstrap mechanics sequence.

---

### TIER 0: Namespace Reconciliation (Do First, Zero Code Until Done)

**0.1 Agree on EventBus channel names across Lead and Alpha.**
Choose one canonical namespace (recommendation: Lead's hierarchical `market.edge.*` / `market.intel.*`). Write it in a single source-of-truth file (`mae_core/market/CHANNELS.md` or as constants in `signal.py`). No wiring code until this is agreed.

---

### TIER 1: Pre-Integration Fixes (Before Any Bootstrap Work)

These are bugs that will cause immediate crashes or permanent silent failures when any integration code runs. They must be fixed and tested before Layer 33 is written.

**1.1 Fix `trade.is_purchase` AttributeError** — `contract_predictor.py:232` and `politician_tracker.py:276` (separate instances of the same root cause). Add `is_purchase` property to `InsiderTrade` dataclass. *Confidence: Very High — confirmed by Lead and Beta independently.*

**1.2 Rebuild `thompson_distributions.json` from seeding logic.** Audit the 22-entry file against the 12-key `source_reliability` in `learning_config.py`. Remove the 10 manually-added entries that have no path back to the config system (including the contradictory `rsi`/`technical_rsi` and `bollinger`/`technical_macd` duplicates). Set `DEFAULT_PRIOR_SCALE = 2`. Run the seeder to produce a clean 12-entry file. Commit this as the known-good starting state.

**1.3 Fix `jobs_30d` baseline** — `job_tracker.py:302-307`. Change `daily_avg = signal.jobs_30d / 30` to `daily_avg = signal.jobs_7d / 7`. The leading indicator chain is broken until this is fixed.

**1.4 Replace all hardcoded `http://localhost:6333`** — `cluster_detector.py:21`, `filing_time_analyzer.py:102`, `contract_predictor.py:32`. Replace with configurable parameter in each constructor. The bootstrap will pass `qdrant_url` from `ctx`.

**1.5 Fix `hash()` non-determinism in Qdrant IDs** — `cluster_detector.py:620`. Replace `abs(hash(cluster_id)) % (10**18)` with `uuid.UUID(cluster_id).int`. Existing duplicate entries in Qdrant cannot be cleaned automatically but new restarts will no longer accumulate duplicates.

**1.6 Fix velocity units** — `velocity_detector.py:132-136`. Change `dt` from seconds to days: `dt = (timestamp - state.last_timestamp).total_seconds() / 86400`. Without this, urgency classification in ConvergenceAlerter is permanently "days" for all daily-frequency signals.

**1.7 Fix timezone handling** — `filing_time_analyzer.py:125-135`. Convert all `filing_datetime` parameters to US/Eastern before comparison against MARKET_OPEN / MARKET_CLOSE boundaries.

**1.8 Replace all `print()` with `logging.getLogger(__name__)`** — all 16 market module files. This is a pre-condition for observability, not a cleanup task.

**1.9 Lower `min_variance` to 0.001** — `thompson_sampler.py:311`. Companion to the `prior_scale=2` change. Prevents future miscalibrated seeds from being locked out of exploration.

**1.10 Change `min_domains` default to 3** — `convergence_alerter.py:89`. Enforce triadic confirmation at the alerter's constructor default. The bootstrap will also pass this explicitly, but the default should be correct.

---

### TIER 2: Signal Format (Central Contract)

**2.1 Create `mae_core/market/signal.py`** with `MarketSignal` dataclass (Lead's design, ~18 fields). This is the shared data contract between all edge detectors and all intelligence consumers. Must be written before any EventBus adapters.

**2.2 Write adapter functions for each source type** — one function per edge detector output type that converts `ClusterSignal`, `FilingTimeSignal`, `ContractPrediction`, etc. into `MarketSignal`. These go in `signal.py` or in a thin `adapters.py` alongside it.

**2.3 Address ContractPredictor double-counting** by assigning it a synthetic domain category (`institutional_synthesis`) that does not overlap with `insider` or `contracts` categories in `domain_categories` map. This is a one-line addition to `convergence_alerter.py`'s `domain_categories` dict and a one-line change in the `ContractPrediction` adapter.

---

### TIER 3: Bootstrap Wiring (Layer 33)

Follow Alpha's Layer 33 design (bootstrap/market.py), with the following additions and modifications from Phase 3 synthesis:

**3.1 Instantiate all market systems** following Alpha's `_instantiate_market_systems()` design. Pass `qdrant_url=ctx.qdrant_url` to any system with a Qdrant dependency.

**3.2 Instantiate `ConvergenceAlerter` with explicit `min_domains=3`** — do not rely on the default.

**3.3 Add `get_statistics()` adapter methods** to ThompsonSampler, ConvergenceAlerter, VelocityDetector (Alpha's Part 10 specification).

**3.4 Complete the K3 fractal repair** — call `ctx.fractal_generator.generate_triad()` to place `market-intelligence-system` as the third child of `organ-cluster-cognitive`. This is the companion to the Qdrant routing fix: both happen in Layer 33, together.

**3.5 Register triadic connections** following Alpha's design, but with a caveat: connections through Qdrant (cluster_detector, filing_time_analyzer, contract_predictor → qdrant_client) should be registered with an explicit note that they are Phase 1 compliance (qdrant_url configurable) rather than Phase 2 compliance (full ApiGateway routing). This prevents the observation lie Alpha's cross-review identified: the registered connections should describe the actual data path.

**3.6 Wire step hooks with deduplication** following Alpha's Surprise 4 design. The step hook should track `_last_convergence_alert` state and suppress re-publication unless direction changes or strength changes by more than 0.1.

**3.7 Wire EndocrineSystem coupling** following Alpha's Part 11 design (bullish convergence → DOPAMINE, bearish → ADRENALINE). The deduplication in step 3.6 is what makes this safe — without deduplication, this becomes a hormone amplifier as my cross-review (Section 5.2) found.

**3.8 Add three stem cell roles** (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST) following Alpha's Part 13 design.

**3.9 Position Layer 33 between external (Layer 31) and audit (Layer 32)** so market systems are included in the audit's connection verification.

---

### TIER 4: OutcomeTracker (Learning Loop Closure)

**4.1 Build OutcomeTracker** — reads `predictions.jsonl`, identifies predictions older than N days, calls `price_fetcher_for_outcomes()` (already exists at `price_fetcher.py:263`), routes success/failure to `ThompsonSampler.update()`. This is approximately 60-80 lines given the constituent functions exist.

**4.2 Add OutcomeTracker to Layer 33** as a 15th market system on ctx (Alpha recommends 14; Lead recommends a TickerResolver as 15th; OutcomeTracker is the more critical gap). OutcomeTracker and TickerResolver should both be added in Tier 4.

**4.3 Implement Bayesian forgetting** in ThompsonSampler via `_apply_forgetting()` method. Connect the decay rates from `learning_config.py` to the Sampler — these are currently dead config. Apply daily via the step hook.

---

### TIER 5: Calibration and Safety (Before Live Operation)

**5.1 Add alert deduplication to ConvergenceAlerter** itself (separate from the step hook deduplication in 3.6). Track `last_alert_direction` and `last_alert_time`. Suppress re-alerting within minimum interval.

**5.2 Cap `self.alerts` list** using `deque(maxlen=1000)`.

**5.3 Increase `CorrelationTracker.min_observations` from 10 to 30.** Statistical adequacy requirement.

**5.4 Add thread locking to ThompsonSampler file writes.** Multi-agent operation will corrupt the distribution file without this.

**5.5 Move `learning_config.py` history log** from `Path(__file__).parent` to `DATA_DIR` for consistency with all other persistence.

**5.6 Use real contact email in `SEC_USER_AGENT`.** Compliance before any production or live testing.

**5.7 Add input validation to `ConvergenceAlerter.record_signal()`** — clamp strength and confidence to [0.0, 1.0].

**5.8 Add `raw_id` reference instead of full `raw_payload` in `MarketSignal`** (from my cross-review Section 5.7). Store a reference to the source record, not a full re-serialized copy. This prevents Qdrant payload bloat and preserves the normalization purpose.

---

### TIER 6: Deferred (Phase 2)

- Full ApiGateway routing for the six market-specific API clients.
- Regime-aware Thompson Sampling (separate distributions per market regime).
- CorrelationTracker deque persistence across restarts.
- ContractPredictor decomposition evaluation (standalone edge detector vs. routed through ConvergenceAlerter).
- Expand `KNOWN_POLITICIANS` beyond 4 entries — consult govtrack.us or ProPublica Congress API for full member list.
- Fix `_identify_politician()` last-name substring matching to full-name matching.
- Expand `_symbol_to_company()` beyond 11 hardcoded mappings — route through the TickerResolver service.
- `discovery_log.jsonl` reader for pattern-library functionality.

---

## Part 4: Net Assessment After Cross-Review

### What the cross-review changed

Three positions shifted meaningfully:

1. **Lead's self-correction on ConvergenceAlerter category-counting**: The boost formula IS correctly implemented. My original finding was wrong about the formula. The `min_domains=2` concern stands; the formula concern does not.

2. **Alpha's K3 fractal repair**: I had no position on fractal placement. The insight that completing the K3 also repairs a pre-existing Law 1 violation changes the implementation framing: fractal repair and Qdrant routing fix are coupled actions.

3. **OutcomeTracker gap**: I had framed this as building a complete missing feedback loop. Lead's finding that `price_fetcher_for_outcomes()` already exists reduces the implementation estimate significantly. The gap is real but smaller than I initially assessed.

### What the cross-review confirmed

Four of my hardest-hitting findings were independently confirmed:

- **`trade.is_purchase` crash bug** (Lead + Beta): This is the clearest pre-fix requirement.
- **`jobs_30d` false spike factory** (Lead confirmed the architectural consequence): The leading indicator chain is broken at its most important node.
- **`thompson_distributions.json` split-brain** (Alpha confirmed, Lead confirmed as Surprise 2): The Bayesian state is not reproducible from the codebase.
- **Alert duplication as behavioral corruption** (Alpha expanded into the endocrine consequence): This became one of the five combined-failure-modes I found in the cross-review.

### What the cross-review exposed that no single investigation found

The EventBus channel name mismatch (Section 5.1 of my cross-review) is the most dangerous new finding. Neither Lead nor Alpha flagged it. It only becomes visible when you hold Lead's channel naming proposal and Alpha's channel naming proposal side-by-side. Zero market intelligence would flow through the integrated system at first boot due to this mismatch — it would bootstrap successfully, log all connections as healthy, and produce nothing.

This is the archetypal reason triadic investigation exists: a failure that is invisible from any single perspective becomes obvious from the combined view.

---

## Summary Statement

My adversarial investigation found the right class of problems. The cross-review corrected one specific finding (ConvergenceAlerter boost formula), refined two (OutcomeTracker scope, `prior_scale=2` arithmetic), and added coupling context to a third (K3 fractal repair and Qdrant routing as paired action). The core findings — overconfident Thompson priors, broken velocity units, false spike detection, non-deterministic Qdrant IDs, alert storms, thread unsafety, and the channel mismatch — all survive intact.

The five combined failure modes I found in the cross-review (Section 5 of beta-cross-review.md) are the most important contribution from Phase 2. They cannot be found by any single investigation and they represent the difference between "three good designs sitting next to each other" and "an integrated system that actually works."
