# Witness Beta Cross-Review: What the Combined Plan Breaks

**Role: Witness Beta — Adversarial Lens**
**Phase: 2 Cross-Review**
**Date: 2026-02-22**
**Reviewing: Lead (Signal Architecture) + Alpha (Bootstrap Integration)**

---

## 1. Reasoning Divergence Points

These are the places where Lead and Alpha reached conclusions that differ from mine, and where the divergence began.

---

### 1.1 Lead concludes the signal normalization problem is "near-complete" — I disagree at step one

**Lead's reasoning path:** Lead read all 16 modules, found that `ConvergenceAlerter.record_signal()` defines the target interface, and concluded the work is "building the single membrane that every data shape collapses into." The framing implies the intelligence layer is ready to receive signals — you just need the adapters.

**Where our reasoning diverged:** My investigation reached the intelligence layer with a different prior: the intelligence layer itself has critical parameter bugs that no adapter can fix. The membrane Lead is designing will successfully deliver signals to a receiver that is miscalibrated in at least five ways before the first real signal arrives. Specifically:

- Lead's `MarketSignal.velocity` field (Part 6.1) is populated "by VelocityDetector after recording; default 0.0" — this is the right design, but the VelocityDetector produces velocities in per-second units while `ConvergenceAlerter` uses a `> 0.1` threshold calibrated for per-day units. Lead identified this gap (Part 10.3) but framed it as "VelocityDetector is Disconnected from ConvergenceAlerter." My finding is stronger: even when connected, the velocity values will be four orders of magnitude too small to trigger the urgency classifier. This is not a wiring gap — it is a unit mismatch that makes the urgency classifier permanently broken regardless of wiring.

- Lead's `MarketSignal` design includes `decay_rate` from `learning_config.decay_rates` (Part 6.1). My investigation found those decay rates are dead code — nothing in any of the five intelligence files reads them. The field will be populated but never acted upon. Lead's design assumes the decay infrastructure is functional.

**Divergence step:** Lead audited the APIs and normalizer design. I audited the receivers. The same system looks different depending on which end you examine first.

---

### 1.2 Alpha concludes Law compliance is achievable with 23 connections — I see a structural gap at the triadic witness level

**Alpha's reasoning path:** Alpha designed three K3 subsystems (market-sensing, market-edge, market-learning) with explicit triadic connections between them. The connection list uses `["auditor", "connection_registry"]` or `["threat_detector", "input_validator"]` as standard witnesses.

**Where our reasoning diverged:** My finding was that the market modules bypass all Mae infrastructure through hardcoded `requests.post("http://localhost:6333/...")` calls. Alpha registered these connections in the connection registry — but the actual data flow will never pass through those registered paths. The registration says "sec_edgar_client -> boundary_membrane, witnesses=[input_validator, threat_detector]." The actual code in `cluster_detector.py`, `filing_time_analyzer.py`, and `contract_predictor.py` makes direct HTTP calls to Qdrant that are invisible to the BoundaryMembrane, invisible to the ConnectionRegistry, and invisible to the witnesses.

**Divergence step:** Alpha designed the registration layer. I examined the execution layer. A registered connection that no actual data flows through is an observer lie — the system believes it is witnessed when the interesting data is flowing on an unregistered path.

Alpha acknowledges this in Part 6: "Full ApiGateway routing is a Phase 2 task." But the triadic connection registration in Part 9 claims these connections as Law 1 compliant NOW. They will not be compliant at the moment they are registered — only after Phase 2.

---

### 1.3 Alpha's Layer 33 log output says "graceful degradation: 0 failed" — this is optimistic

**Alpha's reasoning path:** Alpha identified that missing env vars produce None on ctx, and wraps instantiation in try/except with graceful fallback.

**Where our reasoning diverged:** My investigation found that several constructors make live HTTP calls. `SECEdgarClient.__init__()` does not make network calls, but `PoliticianTracker.__init__()` and `ContractPredictor.__init__()` create `SECEdgarClient()` and `USASpendingClient()` instances at construction time. More critically, `ClusterDetector.__init__()` initializes its Qdrant connection details. If Qdrant is not running during bootstrap, the construction succeeds (no HTTP call at `__init__`) but the first `find_clusters()` call in the step hook will fail silently and return empty results.

Alpha's "graceful degradation: 0 failed" log line will appear even when Qdrant is down, because the failure is deferred to the first operational call. The bootstrap will report success and the system will be silently dead.

**Divergence step:** Alpha examined the `__init__` signatures to determine what fails at construction time. I examined what fails at first use — which happens to be during the step hook in normal operation, not during bootstrap.

---

### 1.4 Alpha places ConvergenceAlerter in market-learning K3 with ThompsonSampler and VelocityDetector — this creates a deduplication problem

**Alpha's reasoning path:** The three systems interact tightly (Velocity feeds ConvergenceAlerter; Thompson weights its output), so grouping them as a K3 subsystem is biologically coherent.

**Where our reasoning diverged:** My finding about alert duplication (Section 3 of my findings) becomes structurally worse with this grouping. Alpha's step hook (Part 7) calls `ctx.convergence_alerter.check_convergence()` on every step. My finding: `check_convergence()` generates a new alert every time conditions are met, with no deduplication. Alpha's design will call this every single step of the simulation. If the convergence condition holds for 100 steps, 100 identical alerts are appended to `self.alerts`, all published to `market.convergence_alert`, and all triggering the endocrine hormone release (Part 11). Mae will receive 100 dopamine doses from a single market condition.

**Divergence step:** Alpha designed the step hook integration. I examined what happens when a step hook calls a non-idempotent method repeatedly.

---

## 2. Agreements — Where Independent Work Converged

These findings arose independently and reinforce each other.

**`trade.is_purchase` crash bug:** Lead identified this at Part 10.1 as a live AttributeError. I identified it as Critical Break #1. Both investigations reached the same conclusion from different entry points (Lead from API output mapping, me from runtime execution path analysis). This agreement raises confidence: it is unambiguously a guaranteed crash.

**`trade.shares_traded` vs `trade.shares` mismatch:** Lead identified this in Part 10.1. I catalogued it in the same context. Both found it independently from reading `politician_tracker.py`.

**VelocityDetector-ConvergenceAlerter disconnection:** Lead identified it in Part 10.3. I identified it in Section 1.4. We described it at different levels of severity — Lead frames it as a wiring gap, I found it is also a units mismatch that persists after wiring. Convergence: the gap exists. Divergence: the severity.

**Congressional trade timing (45-day lag):** Lead identified this in Part 10.5 as a signal decay risk. I identified the timing window problem in `politician_tracker.py` (Section 2.3) as a false positive factory. Different angle, same underlying event timing problem.

**TickerResolver gap:** Lead identified in Part 10.2 that USASpending and SAM.gov have no ticker field. I identified the same problem from the PoliticianTracker's `_symbol_to_company()` being limited to 11 hardcoded symbols (Section 2.3). Both converge on: the contract-domain signals cannot reliably resolve to a ticker.

**Single-category convergence producing neutral output:** Lead identified this in Part 10.6 as a risk for all-institutional signals. I identified the related problem that `min_domains=2` allows fake convergence from correlated signals at the same company. Both agree: the min_domains threshold needs raising.

---

## 3. Gaps — What They Missed That My Investigation Found

**Thompson Sampler prior overconfidence (prior_scale=10):** Neither Lead nor Alpha identified this. Lead uses the Thompson Sampler as a weighting mechanism in the TradeSignal generation pipeline (Part 9.1) and Alpha treats it as a reliable Bayesian source. Both designs assume the Thompson Sampler produces calibrated reliability estimates. My finding: `sec_edgar` starts at Beta(9.5, 0.5) with mean 0.95 and variance 0.004 — tighter than a distribution derived from 100 real observations — before any real data is collected. Lead's `thompson_weighted_confidence` field in TradeSignal will be confidently wrong.

**Thompson Sampler has no forgetting mechanism:** Neither Lead nor Alpha mentioned this. Alpha's learning loop integration (Part 11) feeds successes into `ThompsonSampler.update()`. My finding: the update only increments alpha and beta — they never decay. A signal whose market regime changed 6 months ago will be weighted at full historical confidence indefinitely. Alpha's FRL-to-Thompson feedback loop will accumulate belief in stale signals forever.

**The `jobs_30d` metric is computed from 7-day API data:** Neither Lead nor Alpha identified this. Lead includes `HiringSignal.spike_ratio` in the strength normalization table with formula `min(1.0, spike_ratio / 5.0)`. My finding: `spike_ratio` is computed as `jobs_24h / (jobs_30d / 30)` where `jobs_30d` is actually 7-day data. The denominator is 4x lower than it should be. Lead's normalization formula will regularly produce strength values near 1.0 for ordinary hiring activity.

**`hash()` non-determinism in Qdrant IDs:** Neither Lead nor Alpha identified this. Alpha's connection registration assumes the Qdrant data store is reliable. My finding: `store_cluster_signal()` uses `abs(hash(cluster_id)) % (10**18)` for Qdrant point IDs, and Python's `hash()` is randomized per process. On every restart, the same cluster gets a different ID. The historical wins data that Lead's architecture depends on for feedback loops will accumulate duplicate entries, progressively biasing the ThompsonSampler.

**`KNOWN_POLITICIANS` has only 4 entries:** Neither Lead nor Alpha flagged this. Alpha includes `politician_tracker` as one of the three K3 members in the market-edge subsystem, positioning it as an equal partner with `cluster_detector` and `contract_predictor`. My finding: the politician_tracker is functionally inoperable for 535 of the 539 members of Congress. It will produce zero signals in the wild. Placing it in the K3 is structurally valid but the system built around it will have a hollow node.

**Alert duplication and alert storm:** Alpha's step hook calls `check_convergence()` every step. Neither Lead nor Alpha addressed the deduplication gap. My finding: without a suppression mechanism, each step that evaluates a convergence-active state produces a new alert. At 30 steps per run, a persistent convergence condition produces 30 alerts, 30 endocrine doses, and 30 EventBus publications — all from a single sustained market condition.

**`_prune_old_signals()` called on every insert:** This performance problem is not mentioned by Lead or Alpha. Both assume the ConvergenceAlerter can serve as the synchronous heart of the pipeline. My finding: at high EventBus volume, the O(n*domains) prune on every `record_signal()` call will introduce latency in the EventBus callback path, blocking the entire bus while pruning.

**The `thompson_distributions.json` split-brain:** Neither Lead nor Alpha examined the current state of the distribution file. Lead treats Thompson distributions as the source of truth for signal weighting. My finding: the file has 22 entries but `learning_config.py` only seeds 12. The 10 extra entries were manually added and have contradictory values (`rsi` at mean=0.167 vs `technical_rsi` at mean=0.857 — the same signal type with opposite reliability beliefs). A fresh deployment will not reproduce the current file state, and the current file state already contains duplicated conflicting beliefs.

**`update_config()` race condition and wrong path:** Alpha's bootstrap creates a shared `ctx.learning_config` (implicitly — it's a module-level dict). Multiple agents calling `update_config()` simultaneously will produce dict corruption. The function also writes its history log to `Path(__file__).parent / "config_history.jsonl"` — inside the market module directory, not in `data/market/` where everything else persists. This file will be in the wrong place and potentially commited to git.

---

## 4. Surprises — Findings That Changed My Thinking

**The ConvergenceAlerter's domain categories ARE correctly deduplicating by category, not by domain name.** I initially assessed the confidence boost formula as potentially double-counting. On closer re-reading, `categories_seen` uses the `domain_categories` map, so five "market" signals only contribute 1 to `cross_domain_count`. Lead's analysis of this was correct. I had marked this as a risk in my initial read; the cross-review forced me to recheck. The boost formula is more carefully designed than I gave it credit for. This means the formula concern I raised (that the strength term is doing no useful work) remains, but the category-counting concern is resolved.

**Lead's channel naming proposal (`market.signal.raw`, `market.signal.scored`, etc.) is more granular than Alpha's.** Alpha proposes channels like `market.cluster_signal`, `market.politician_trade`, `market.contract_prediction`. Lead proposes channels like `market.edge.cluster_detected`, `market.edge.correlation_found`. These two channel namespaces are incompatible — Alpha's bootstrap EventBus wiring would register different channel names than Lead's architecture specifies. If both plans are implemented independently, the publish and subscribe sides will miss each other. This is a coordination gap that neither document flags.

**Alpha's fractal placement of market-intelligence-system inside organ-cluster-cognitive is actually biologically coherent in a way I had not considered.** My adversarial framing was skeptical of the fractal integration. But Alpha's reasoning — "market intelligence IS cognitive — it senses the external information environment and reasons about it" — holds up. The biological coherence means the fractal constraints are satisfied with minimal forcing. The concern I flagged about hollow K3 nodes (a 4-member list left over from the housestockwatcher/usa_spending/sam_gov/correlation_tracker non-K3 residuals) is real but manageable.

**Lead's observation that `ContractPredictor` is already a domain-specific version of `ConvergenceAlerter`** (Part 2.4) was not something I had articulated. My investigation found bugs in ContractPredictor but did not notice the architectural redundancy. This has a real implication: when both are wired to the EventBus and both run on the same contract signals, they may generate duplicate convergence conclusions via different code paths. The system could arrive at the same conclusion twice through different routes and interpret it as independent confirmation.

---

## 5. What Breaks — New Failure Modes From the Combined Plan

These failure modes only become visible when Lead's signal design and Alpha's bootstrap plan are combined. Neither document surfaces them alone.

---

### 5.1 EventBus Channel Name Mismatch — Lead and Alpha Use Different Namespaces

**Lead's proposed channels** (Part 7.1):
```
market.edge.cluster_detected
market.edge.correlation_found
market.edge.filing_anomaly
market.edge.contract_predicted
market.intel.velocity_anomaly
market.intel.convergence
market.intel.actionable
```

**Alpha's proposed channels** (Part 9 connection list):
```
market.cluster_signal
market.politician_trade
market.contract_prediction
market.filing_signal
market.convergence_alert
market.thompson_stats
market.velocity_anomaly
```

These are different names. Alpha's bootstrap wires `cluster_detector -> event_bus` with `channel="market.cluster_signal"`. Lead's architecture has `ConvergenceAlerter` subscribing to `market.edge.cluster_detected`. The publish and subscribe sides use different channel strings. Zero signals will flow from edge detectors to the ConvergenceAlerter until this is reconciled. The system will bootstrap successfully, log all connections as healthy, and produce no market intelligence output.

**Severity: COMPLETE INTEGRATION FAILURE** — the two halves of the signal pipeline will not connect.

---

### 5.2 The Step Hook Calls a Non-Idempotent Method on Every Step

Alpha's step hook (Part 7) calls `ctx.convergence_alerter.check_convergence()` on every simulation step. Lead's ConvergenceAlerter design (Part 3.1) shows that `check_convergence()` generates and appends new alerts every call where conditions are met, with no idempotency guard.

Combined effect: if a convergence condition is sustained (which is the normal case — a market event lasts days, not steps), and if Alpha's step hook fires 30 times during a run, and if Lead's endocrine wiring (Alpha Part 11) calls `ctx.endocrine.release_hormone(DOPAMINE, ...)` on every `market.convergence_alert` event:

Mae receives 30 dopamine pulses from a single market observation. The endocrine system accumulates hormone levels that bear no relationship to the number of actual market events — only to how many simulation steps occurred while a condition held. Mae's behavioral state will be dominated by phantom hormone accumulation from loop repetition, not from information.

**Severity: BEHAVIORAL CORRUPTION** — the endocrine feedback loop, which Alpha presents as a feature, becomes a hormone amplifier.

---

### 5.3 The Feedback Loop Has a Missing Leg: No OutcomeTracker Exists

Lead's feedback loop design (Parts 8 and 9.1) is the most complete piece of architecture in either document. It requires:
1. Prediction stored in `predictions.jsonl`
2. N days later, PriceFetcher checks outcome
3. `ThompsonSampler.update(signal_id, success, regime)` called
4. Beta distribution shifts

The component that executes step 2 and 3 — an OutcomeTracker — **does not exist** in any of the 16 market modules, in the bootstrap plan, or in Alpha's Layer 33 design. Alpha's Part 8 includes 14 instantiable systems; OutcomeTracker is not among them. Lead's architecture assumes OutcomeTracker will exist but does not specify where it comes from.

Without OutcomeTracker, `predictions.jsonl` fills with predictions that are never evaluated. `ThompsonSampler.update()` is never called from real outcomes. The Bayesian distributions frozen at their seeded values. The system never learns.

Alpha's FRL-to-Thompson feedback loop (Part 11) partially compensates — FRL rewards could trigger Thompson updates indirectly — but this is not the same as measuring whether a specific market signal's predicted price direction was actually correct.

**Severity: LEARNING NEVER HAPPENS** — the system's core differentiator (Bayesian signal reliability learning) is architecturally specified but not instantiated.

---

### 5.4 ContractPredictor and ConvergenceAlerter Will Double-Count Defense Sector Signals

My finding from Section 4 of my own document: ContractPredictor is already a domain-specific ConvergenceAlerter. Lead identified this independently (Part 2.4) but did not trace the failure mode.

In Alpha's K3 design, both `contract_predictor` and `convergence_alerter` are wired to receive signals from `cluster_detector` and `politician_tracker`. ContractPredictor synthesizes those signals into a `ContractPrediction` with a confidence score. That `ContractPrediction` is then published on the EventBus as a signal — and Lead's adapter table (Part 6.2) maps `ContractPrediction` → `domain="contracts"`.

`ConvergenceAlerter` then receives the raw insider signals AND the synthesized ContractPrediction derived from those same signals. The defense convergence fires based on signals that are, in part, derived from themselves. A single insider buy at a defense contractor is counted once as an "insider" domain signal and again inside the ContractPrediction as a "contracts" domain signal. Cross-domain count becomes 2 (insider + contracts) when the independent evidence is 1 (the buy).

Lead's `min_domains` mitigation (raising to 3) does not solve this — it only raises the bar. The same double-counting structure inflates confidence regardless of the threshold.

**Severity: SYSTEMATIC CONFIDENCE INFLATION** for defense sector signals. The system's highest-confidence domain is structurally double-counting.

---

### 5.5 Alpha's `get_statistics` Adapter Aliases Break if the Underlying Method Is Called During Shutdown

Alpha recommends (Part 10) adding `get_statistics = get_stats` as an alias for HolonProxy delegation. The HolonProxy `heal()` capability checks system health and calls `reset()` if degraded. If a market system is "healing" (resetting state) while `get_statistics()` is being called by the AwarenessPulse, and if `reset()` clears the internal data structures that `get_statistics()` is iterating, the result is a race condition on the market system's internal state.

This was identified in Section 3.3 of my findings as a general thread-safety gap. Combined with Alpha's proxy injection design, every `sense()` call via HolonProxy becomes a potential race with the proxy's own `heal()` call. Alpha's "non-breaking, additive" adapter additions are non-breaking only in a single-threaded context.

**Severity: INTERMITTENT STATE CORRUPTION** — will appear as occasional spurious statistics values or empty results during heal cycles, not as a clean crash.

---

### 5.6 `DATA_DIR` Path Resolution Assumes a Specific Working Directory

Alpha's Part 8 notes that `ThompsonSampler` and related modules use `Path(__file__).resolve().parents[3]` to find `data/market/`. This is documented in MEMORY.md as a fixed path.

Lead's architecture adds a `SignalWriter` service (identified but not specified) that must also write to `data/market/`. Alpha's bootstrap plan instantiates `ClusterDetector`, which writes to Qdrant (not `data/market/`), but `ThompsonSampler._save_distributions()` writes to `data/market/thompson_distributions.json`.

The problem: when multiple market systems are writing to `data/market/` simultaneously — `ThompsonSampler` writing distributions, `ConvergenceAlerter` (if given persistence), and any future `OutcomeTracker` writing to `predictions.jsonl` and `outcomes.jsonl` — and all writes are non-atomic (write_text is not atomic), concurrent writes can produce partial files.

My finding (Section 3.3, Section 5.3) addressed this for ThompsonSampler alone. Combined with Lead's full pipeline design that adds multiple new writers to the same directory, and Alpha's multi-agent step hooks running concurrently, the `data/market/` directory becomes a shared mutable resource with no coordination.

**Severity: DATA CORRUPTION under concurrent operation** — the scenario Alpha explicitly designs for (multiple market agents running simultaneously).

---

### 5.7 Lead's `MarketSignal.raw_payload` Will Cause the Qdrant Payload to Exceed Size Limits

Lead's `MarketSignal` dataclass includes `raw_payload: dict` — the original dataclass as dict "for audit and re-processing." The `ClusterDetector` already stores signals in Qdrant. If the `SignalWriter` stores the full `MarketSignal` including `raw_payload`, SEC EDGAR Form 4 filings (which include company info, transaction history tables, and footnotes) can be several kilobytes of parsed content.

Qdrant's recommended payload size limit per point is under 1MB, but with hundreds of signals per day, the `midge_signals` collection grows rapidly. More concretely: the `raw_payload` field contains a re-serialized dataclass dict, which contains the same data that Lead recommends normalizing away from. The normalizer exists to remove the raw heterogeneous data — storing the raw payload defeats the normalization purpose and adds storage burden.

**Severity: STORAGE BLOAT and normalization purpose undermined.** Recommend storing a `raw_id` reference rather than the full payload.

---

### 5.8 Alpha's Layer 33 Runs Before the Audit But After Seal — Audit Will Count Market Systems

Alpha notes (Part 14) that adding 14 market systems pushes `systems` to ~99 and `holons` to ~124, both well above the audit thresholds of 75. However: the audit at Layer 32 validates `_EXPECTED_SYSTEMS` and `_EXPECTED_CONNECTIONS` hardcoded lists (not just minimum counts). Adding Layer 33 BEFORE the audit means the audit runs against a world that includes market systems but `test_integration.py` still expects the old counts (2425 tests, 85 systems, 313 connections).

Alpha identifies the document parity requirement (Part 14) but notes it as a task — not as a test failure. The test suite will fail on the first run after Layer 33 is added unless `test_integration.py` is updated simultaneously. This is a zero-regression guarantee violation at the moment of integration.

**Severity: TEST SUITE FAILURE on first run** — expected by Alpha as "update test_integration.py" but not flagged as a risk that must be done atomically with the bootstrap change.

---

## Summary

The combined Lead + Alpha plan is architecturally coherent and biologically complete. The designs are complementary and mutually reinforcing. Both investigators did excellent independent work.

The cross-review reveals five new failure modes that only emerge from the combination:

| # | Failure | Severity |
|---|---------|----------|
| 5.1 | EventBus channel name mismatch between Lead's design and Alpha's wiring | COMPLETE PIPELINE FAILURE |
| 5.2 | Non-idempotent step hook causes hormone amplifier from loop repetition | BEHAVIORAL CORRUPTION |
| 5.3 | OutcomeTracker is architecturally required but not instantiated in Layer 33 | LEARNING NEVER HAPPENS |
| 5.4 | ContractPredictor + ConvergenceAlerter double-count defense signals | CONFIDENCE INFLATION |
| 5.5 | HolonProxy heal() races with get_statistics() in multi-agent context | INTERMITTENT CORRUPTION |
| 5.6 | Multiple market writers to data/market/ with no coordination | DATA CORRUPTION |
| 5.7 | raw_payload in MarketSignal bloats Qdrant and undermines normalization | STORAGE + DESIGN CONFLICT |
| 5.8 | Test suite fails atomically when Layer 33 is added pre-audit | IMMEDIATE TEST FAILURE |

**The single most dangerous gap not found by either investigator: there is no OutcomeTracker (5.3).** The feedback loop is the design's core value claim. Without it, MIDGE is a static pattern matcher with borrowed confidence scores that never update from reality.

**The single most dangerous implementation gap: the EventBus channel name mismatch (5.1).** Zero signals will flow from edge detectors to the ConvergenceAlerter unless Lead and Alpha reconcile their channel naming before any code is written.

**The single most dangerous calibration gap (from my original findings, confirmed by cross-review): `DEFAULT_PRIOR_SCALE=10` in ThompsonSampler.** Both Lead and Alpha build architectures that trust the Thompson Sampler as a reliable weighting authority. It is not — it starts with fake confidence that looks real.
