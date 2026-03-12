# Team 1 Findings: System Liveness Census
## Date: 2026-03-12
## Researcher: Team Member 1

---

## Methodology

I traced every system from bootstrap instantiation through step hooks and EventBus subscribers to determine whether meaningful output flows somewhere else in the organism. Four categories:

- **ALIVE**: Produces output that another system demonstrably consumes in the live step loop
- **DORMANT**: Constructed and passed around, never called during daemon operation
- **DECORATIVE**: Called each step, output stored on ctx or published to bus, but nothing downstream reads it
- **ZOMBIE**: Called but the channel it depends on has no producer — permanently starved

---

## Battle-Tested Approaches

### The ALIVE Core — What Actually Runs

These systems form MIDGE's functioning backbone. Each has a clear call chain from step hook to output consumed downstream.

**ALIVE — Primary market intelligence pipeline:**

| System | Call site | Cadence | Consumer |
|--------|-----------|---------|----------|
| `ConvergenceAlerter` | `_market_sense_hook()` | Every step | Paper trade gate, Alpaca, PatternMemory, endocrine coupling, 15+ EventBus subscribers |
| `ThompsonSampler` | `_market_sense_hook()` every 10 steps | /10 | `get_stats()` → bus, forgetting at /200 |
| `HypothesisEngine` | `_market_sense_hook()` | Every step | Calls `step()` → manages probation/active/retired cycle |
| `RegimeClassifier` | `_market_sense_hook()` via `_get_regime()` | Every 10/200 | Thompson forgetting cadence selector, convergence alerter |
| `VelocityDetector` | `_market_sense_hook()` | Every 50 steps | Bus publishes `CH_VELOCITY_ANOMALY` → bio systems |
| `DeepAnalyst` | `_run_slow_cadence_ops()` | Every 500 steps | `ctx.inevitabilities`, paper trade validator, PatternMemory, JSONL |
| `LagCorrelationAnalyzer` | `_run_slow_cadence_ops()` | Every 500 steps | `ConvergenceAlerter.set_lag_findings()`, WorldModel auto-edge discovery |
| `GrangerAnalyzer` | `_run_slow_cadence_ops()` | Every 500 steps | WorldModel auto-edge discovery |
| `PostMortemReviewer` | `_run_slow_cadence_ops()` | Every 500 steps | Thompson updates, `post_mortem_insights.json` |
| `ThompsonCalibrator` | `_run_slow_cadence_ops()` | Every 1000 steps | `calibrate()` adjusts Thompson distributions |
| `CascadeTracker` | `_run_slow_cadence_ops()` + EventBus | Every 500 + signal-triggered | `expire_stale()`, WorldModel feedback |
| `WorldModel` | Wired into `ConvergenceAlerter` | Per-convergence | `ripple_effects` on every alert, causal watch subscriber |
| `PatternWatcher` | `_sensing_step_with_advisory()` | Every 10 steps | Pattern stacks → outcome collector → Thompson |
| `ExcavationDaemon` | `_run_slow_cadence_ops()` | Every 5000 steps | Fingerprints → PatternLibrary templates |
| `ActiveTracker` | `_sensing_step_with_advisory()` | Every 20 steps | Force-grades → Thompson feedback |
| `OutcomeCollector` | `_evaluate_outcomes()` in sensing hook | Every 200 steps | Thompson updates (feedback loop closure) |
| `OctopusColony` | `_run_octopus_dispatch()` | Every 20 steps | `investigate_partial` + `situation_check` tasks |
| `DrawdownMonitor` | Paper trade gate | Each trade attempt | Circuit breaker (halts trading) |
| `SystemHealthMonitor` | Multiple try/except blocks | Per operation | `record_error()`/`record_success()` |
| `SelfMonitor` | Paper trade gate | Each trade attempt | Behavioral anomaly suppression |
| `RawDataAnalyst` | `_run_raw_analyst()` | Every 100 steps | Enriched signals → convergence engine |
| `PatternLibrary` | Via PatternWatcher, ExcavationDaemon | Inherited | Template storage and query |
| `DriftDetector` | `_run_drift_detector()` | Every 50 steps | Bus publish `market.intel.drift_detected` |
| `MotifDetector` | Step hook | Every 100 steps | `update()` → signals → convergence alerter |
| `StreamingAnomalyDetector` | Step hook | Every 100 steps | Anomaly score → convergence alerter signal |
| `MarketSensingHook` | Step hook | Every 25 steps (fetch cadence) | 35 data sources → convergence alerter |
| `BacktestScheduler` | `_run_slow_cadence_ops()` | Every 5000 steps | `check_and_schedule()` → HypothesisEngine |
| `StepTimer` | Wraps tracked operations | Per tracked op | Performance tracking, SystemHealthMonitor input |
| `ConsolidationEngine` | `_run_step_consolidation()` | Every 5000 steps | Prunes Thompson distributions + hypothesis archive |
| `SomaticAnticipation` | `_run_step_somatic()` | Every 25 steps | `check_anticipation()` → logs events |
| `PatternArchetypeEngine` | `_run_step_archetypes()` | Every 100 steps | `scan_for_archetypes()` per watchlist ticker |
| `PatternCompletionEngine` | `_run_step_pattern_completion()` | Every 100 steps | `review_partial_matches()` |
| `AbsenceMonitor` | `_run_step_absence()` | Every 100 steps | `check_absences()` → convergence signals |
| `CorrelationTracker` | `_run_step_correlation()` | Every 200 steps | `detect_cross_domain_anomalies()` |
| `CatalystCalendar` | Sensing hook step | Every 200 steps | `refresh()` — data fed to ConvergenceAlerter via `_catalyst_calendar` ref |
| `PortfolioTracker` | `_run_step_portfolio()` | Every 50 steps | Exit signals → convergence alerter |
| `FinnhubWebSocket` | Background thread started in sensing hook | Continuous | Realtime signals → sensing pipeline |
| `KellyPositionSizer` | Step hook | Every 50 steps | `recommend()` → `market.intel.kelly_sizing` bus |
| `PatternMemory` (Qdrant) | Step hook + DeepAnalyst | Per convergence + /500 | Semantic memory (when Qdrant available) |

**ALIVE — API clients that produce signals (rotated through sensing hook):**
All 35 sources in `SOURCE_ROTATION` are ALIVE in the fetch rotation: SEC EDGAR, yfinance, Congressional, Senate, job tracker, USASpending, SAM.gov, ApeWisdom, FINRA, FRED, Finnhub, COT, StockTwits, VIX, Google Trends, TA indicators, CoinGecko, CoinCap, OpenInsider, EDGAR enhanced, FinViz, Economic Calendar, Massive/Polygon, EIA, Congress.gov, Yahoo RSS, USDA, Binance funding, Kalshi, session sweep, form8k_sentiment, cluster_detector, politician_tracker, contract_predictor, social_text_analyzer, fractal_resonance, order_flow.

---

## Novel Approaches

### Critical Discovery: CH_PREDICTION_RESULT Has No Producer

**This is the most structurally significant finding in this census.**

The channel `market.sensing.prediction_result` (`CH_PREDICTION_RESULT`) is subscribed to by **10 bio-systems** across the wiring files (ArousalRegulator, NociceptionSystem, MetacognitionMonitor, HAVEN, LymphaticSystem, SenescenceManager, VestibularSystem, ProprioceptionSystem, Stigmergy, and more). Every one of these bio-systems depends on this channel firing to do its market job.

Searching the entire `mae_core/` codebase for any `publish()` call with `prediction_result` yields **zero results in production code**. The channel is published only in test files (`tests/test_bio_market_wiring.py`, `tests/test_bio_market_wiring_extended.py`).

The `OutcomeCollector.evaluate()` method grades predictions and updates Thompson, but it never publishes `CH_PREDICTION_RESULT` to the bus. The `OutcomeTracker` (separate class, also on ctx) also never publishes it.

**Impact:** Every bio-system callback registered on `CH_PREDICTION_RESULT` is permanently starved. The organism's interoceptive feedback loop from prediction outcomes to biological state is severed. This makes approximately 10 bio-system wirings ZOMBIE — subscribed but never triggered.

**Affected systems:** ArousalRegulator (reward signal), NociceptionSystem (prediction failure pain), MetacognitionMonitor (calibration), HAVEN (source trust recovery), LymphaticSystem (failed prediction cleanup), SenescenceManager (outcome tracker activity), VestibularSystem (prediction accuracy metric), ProprioceptionSystem (outcome tracker health), Stigmergy (success/danger markers), StimulusReactive (general)

---

## Emerging Approaches

### Bio-System Liveness Classification

With the above in mind, here is the precise classification of each bio-system wiring:

**ALIVE bio-systems** (their subscribed channels actually fire):

| System | Channel | Producer | What it does |
|--------|---------|---------|--------------|
| `EmotionalSystem` | `CH_CONVERGENCE`, `CH_DECEPTION_DETECTED`, `CH_DUAL_CONFIRMATION` | ConvergenceAlerter, DeceptionDetector | Adjusts `_surprise_boost` / `_fear_reinforcement` |
| `HomeostasisRegulator` | `CH_CONVERGENCE`, `CH_VELOCITY_ANOMALY` | Same | Updates `threat_level`, `energy_level`, `processing_load` setpoints |
| `InhibitionSystem` | `CH_CONVERGENCE`, `CH_DECEPTION_DETECTED` | Same | Adjusts `_market_caution` on ctx |
| `DigestiveSystem` | `CH_CONVERGENCE`, `CH_PARTIAL_CONVERGENCE` | ConvergenceAlerter | Calls `ingest()` — energy budget gating |
| `CirculatorySystem` | `CH_CONVERGENCE`, `CH_VELOCITY_ANOMALY` | Same | Calls `request_resource()` — attention allocation |
| `LymphaticSystem` | `CH_DECEPTION_DETECTED` | DeceptionDetector | Calls `collect_waste()` — orphan subscription cleanup (deception side only; prediction failure side starved) |
| `Microbiome` | `CH_CONVERGENCE`, `CH_VELOCITY_ANOMALY`, `CH_PATTERN_STACK_DETECTED` | Various | Calls `process_input()` on three strains |
| `RenalFilter` | `CH_CONVERGENCE`, `CH_DECEPTION_DETECTED` | Same | `add_toxin_pattern()` + `filter_item()` |
| `SenescenceManager` | `CH_CONVERGENCE` | ConvergenceAlerter | Reports `convergence_alerter` activity (outcome_tracker side starved) |
| `MorphogenesisCoordinator` | `CH_PARTIAL_CONVERGENCE`, `CH_HYPOTHESIS_DISCOVERED` | ConvergenceAlerter, HypothesisEngine | Spawns investigation organs on partial overflow |
| `ReproductiveSystem` | `CH_CONVERGENCE`, `CH_PARTIAL_CONVERGENCE` | Same | Accumulates `ctx._market_activity_pressure` |
| `PearlDefense` | `CH_DECEPTION_DETECTED` | DeceptionDetector | Quarantine validation for suspicious sources |
| `RespiratorySystem` | `CH_CONVERGENCE`, `CH_VELOCITY_ANOMALY` | Same | Calls `consume_oxygen()` |
| `ThermoregulationSystem` | `CH_CONVERGENCE`, `CH_VELOCITY_ANOMALY` | Same | Reports `report_activity()` |
| `PredictiveField` | `CH_CONVERGENCE` | ConvergenceAlerter | `update_agent_state()` — spatial prediction field |
| `QuorumSpace` | `CH_CONVERGENCE`, `CH_PATTERN_STACK_DETECTED`, `CH_DUAL_CONFIRMATION` | Various | `deposit_signal()` — organism-level vote |
| `CircadianRhythm` | `CH_PHASE_CHANGE` (self-generated) | Internal | Sets `ctx._circadian_activity` multiplier |
| `EnergyReserve` | `CH_CONVERGENCE`, `CH_PHASE_CHANGE` | Same | `release()` / `store()` on convergence and phase |
| `Stigmergy` | `CH_CONVERGENCE` only (half-alive) | ConvergenceAlerter | Convergence trail markers work; success/danger markers starved |
| `ThreatDetector` | `CH_DECEPTION_DETECTED` | DeceptionDetector | Quill registration, sacrificeable components |
| `HAVEN` | `CH_DECEPTION_DETECTED` only (half-alive) | DeceptionDetector | Flag accumulation works; trust recovery starved (no prediction result) |
| `CuriosityDrive` | `CH_PARTIAL_CONVERGENCE`, `CH_HYPOTHESIS_DISCOVERED`, `CH_PATTERN_STACK_DETECTED` | Various | Exploration bonus adjustments |
| `MemoryConsolidator` | `CH_PHASE_CHANGE` | CircadianRhythm | Runs hypothesis engine on CONSOLIDATION, excavation on REST |
| `CollectiveDreamPlanner` | `CH_CONVERGENCE` | ConvergenceAlerter | Nudges agent expertise weights |

**ZOMBIE bio-systems** (subscribed channel has no producer in production code):

| System | Dead channel | Impact |
|--------|-------------|--------|
| `ArousalRegulator` | `CH_PREDICTION_RESULT` (part of wiring) | No reward signal from outcomes — arousal level only moves on convergence, never on wins/losses |
| `NociceptionSystem` | `CH_PREDICTION_RESULT` | Prediction failure pain never fires — organism learns pain from deception but not from being wrong |
| `MetacognitionMonitor` | `CH_PREDICTION_RESULT` | Confidence calibration records are never updated from real outcomes |
| `HAVEN` | `CH_PREDICTION_RESULT` (recovery side) | Source trust flags accumulate but never reduce on successful predictions |
| `LymphaticSystem` | `CH_PREDICTION_RESULT` (failure side) | Wrong predictions never trigger waste collection |
| `SenescenceManager` | `CH_PREDICTION_RESULT` | `outcome_tracker` activity never reported — wears aging signal |
| `VestibularSystem` | `CH_PREDICTION_RESULT` | Prediction accuracy metric stream is empty — vertigo detection blind |
| `ProprioceptionSystem` | `CH_PREDICTION_RESULT` | Outcome tracker body map position never updated |
| `Stigmergy` | `CH_PREDICTION_RESULT` | Success/DANGER markers never deposited — trail map lacks outcome memory |

**DECORATIVE bio-systems** (called, but output not consumed by anything meaningful):

| System | What it does | Why decorative |
|--------|-------------|---------------|
| `ReproductiveSystem` | Accumulates `ctx._market_activity_pressure` | `ctx._consume_market_pressure()` is defined but never called from any step hook. The pressure never drives agent spawn decisions. |
| `VestibularSystem` | Calls `report_metric("convergence_rate", ...)` | No downstream system reads vestibular state. There's no hook that checks if vertigo is firing. |
| `ThermoregulationSystem` | Calls `report_activity()` on convergence | No downstream system reads temperature output. No throttling happens based on heat state. |
| `RespiratorySystem` | Calls `consume_oxygen()` | O2 depletion state is never read by any system to throttle sensing. |
| `PredictiveField` | Updates spatial prediction field | No agent reads the field gradient for coordination. The field accumulates but nobody navigates it. |
| `CirculatorySystem` | `request_resource()` for attention | No system checks whether the circulatory resource request was granted or denied. |
| `DigestiveSystem` | `ingest()` data as nutrients | Energy budget updated but no downstream system gates on digestive capacity. |
| `MorphogenesisCoordinator` | `handle_novel_problem()` on partial overflow | Spawns investigation organs but no system reads or acts on whatever MorphogenesisCoordinator produces. |
| `CollectiveDreamPlanner` | Nudges agent `expertise` weights | Expertise weights affect dream planning only; no feedback path to market sensing or convergence. |
| `EnergyReserve` | `release()` / `store()` | API budget metaphor — but `ctx.energy_reserve` state is never read by rate limiters or API callers. |

**SPECIAL CASE — DORMANT:**

| System | Status | Why |
|--------|--------|-----|
| `InhabitantScheduler` | Dormant | Started as a daemon thread (`_sched.start()`) but dispatches wall-clock events. It runs, but its events don't trigger market operations — it's a wall clock disconnected from the step loop. |
| `GovernanceLogger` | Dormant | Constructed with event_bus reference. Only logs governance events passively. No event has been shown to trigger governance logging from market operations. |
| `ResourceGovernor` | Partially alive | Endocrine cortisol coupling registered (`register_resource_governor`). But the governor's actual budget-enforcement logic is never called from sensing or step hooks. It receives signals but nothing reads its API budget decisions. |

---

## Gaps and Unknowns

### 1. Fractal Resonance and Order Flow — Alive via rotation, not step hooks

`FractalResonanceDetector` and `OrderFlowDetector` are ALIVE but through a different path than expected. They are called via `fetch_fractal_resonance()` and `fetch_order_flow()` in `sensing_reactive.py` as part of the source rotation in `MarketSensingHook`. Each rotation slot calls these fetch functions which call `.detect()` on the detectors directly. The detectors are not called from `_run_step_*` methods — they're part of the signal fetch rotation. Evidence: `sensing_reactive.py` lines 224-228.

### 2. `_consume_market_pressure` call site unknown

`ctx._consume_market_pressure` is defined in `_wire_reproductive()` but I could not find where it's called in any step hook. If there is a call site I missed, ReproductiveSystem may be more alive than classified. Requires a grep of step hooks for `_consume_market_pressure`.

### 3. Kalshi client — ALIVE but incomplete

`KalshiMarketClient` is constructed and passed to the sensing hook. It appears in `SOURCE_ROTATION`. However the Kalshi SDK was noted as "not yet verified against demo env" in MEMORY.md. The client may be producing empty results rather than true signals, making it effectively a zombie despite being wired.

### 4. `SocialTextAnalyzer` — depends on RawStore having StockTwits data

`SocialTextAnalyzer` reads from SQLite (RawStore). If StockTwits has not populated the raw store in the current session, `analyze()` returns empty. This is condition-dependent liveness — alive when the raw store has recent data, empty otherwise.

### 5. Bio-systems that update state but state is not readable by other systems

Several bio-systems (ThermoregulationSystem, RespiratorySystem, VestibularSystem) update internal state via their APIs, but there is no standardized way for the market intelligence pipeline to READ that state and make decisions based on it. The wiring is one-directional — market events flow into bio systems — but the biological homeostatic response never flows back to market behavior.

---

## Synthesis

### The Central Inevitability: A One-Way Mirror

MIDGE's market intelligence pipeline is a functional sensing and inference engine. The ALIVE systems genuinely converge, learn, and adapt. The pattern detection, Thompson sampling, hypothesis engine, DeepAnalyst, and CascadeTracker form a coherent and increasingly capable intelligence layer.

But the bio-system integration — 29 of 30 systems "wired" — is mostly theatrical. The market events flow INTO the biological systems (causing hormone shifts, consuming oxygen, depositing pheromones), but the biological state flowing BACK to market intelligence is almost completely absent.

Specifically:

**What works (market → bio):** Convergence alerts fire hormones, raise threat levels, activate threat detectors, deposit stigmergy markers, trigger quorum votes. This layer is genuinely wired and active.

**What's broken (bio → market, and bio → bio):** The organism's prediction outcomes never return to the bio layer. Nine bio-systems have callbacks on `CH_PREDICTION_RESULT` that will never fire — the OutcomeCollector grades predictions and updates Thompson distributions silently, without broadcasting the result to the EventBus. No `bus.publish(CH_PREDICTION_RESULT, ...)` call exists in any production code. This single missing publish severs the outcome-feedback half of the bio-market loop.

**What's decorative:** Ten bio-systems update internal state that nobody reads. The ReproductiveSystem accumulates market pressure that never drives spawning. The VestibularSystem detects instability that nobody acts on. The ThermoregulationSystem feels heat that triggers no throttling. These are sensors with no effectors.

### Ranked Opportunities

**Priority 1 — Single fix, 9 systems wake up immediately:**
Add `bus.publish(CH_PREDICTION_RESULT, {...})` in `OutcomeCollector.evaluate()` when an outcome is graded. This single trivial addition activates ArousalRegulator, NociceptionSystem, MetacognitionMonitor, HAVEN recovery, LymphaticSystem cleanup, SenescenceManager outcome tracking, VestibularSystem accuracy metric, ProprioceptionSystem body map, and Stigmergy outcome markers. Effort: trivial (5 lines).

**Priority 2 — Wire `_consume_market_pressure` into agent lifecycle:**
ReproductiveSystem's pressure accumulator is built and accumulating — it just needs a caller that passes the pressure to `repro.update_metrics(pressure)`. If that wiring exists, the population-scaling loop closes. Effort: trivial.

**Priority 3 — Read bio-state back into market decisions:**
The current architecture assumes market events flow into bio systems unidirectionally. For the bio systems to matter, their homeostatic states need to influence market behavior:
- RespiratorySystem O2 → throttle sensing concurrency
- ThermoregulationSystem temperature → scale fetch cadence
- VestibularSystem vertigo → flag instability to convergence alerter
- CirculatorySystem resource grants → prioritize high-urgency tickers
Effort: moderate (each requires one new reader in a step hook or gate).

**Priority 4 — InhabitantScheduler real jobs:**
The scheduler runs on a wall clock but dispatches to nobody. Give it real cadenced jobs: refresh `CatalystCalendar` on market open, trigger `PostMortemReviewer.review()` at end of market session, run `ConsolidationEngine.consolidate()` during off-hours. Converts a dormant system into a genuine biological circadian governor. Effort: moderate.

### The Internal Inevitability

MIDGE is looking outward for market inevitabilities while an internal inevitability sits unrealized: the prediction-outcome feedback loop is broken precisely where it would most enrich the organism's biological self-awareness. The OutcomeCollector knows when MIDGE was right or wrong. This knowledge is already being used to update Thompson distributions. The only missing step is broadcasting that knowledge to the EventBus so the organism's body can feel the result of its beliefs. When that loop closes, the organism will hurt when it's wrong, feel reward when it's right, calibrate its confidence from lived experience, and deposit outcome memories in pheromone trails — all from one five-line addition.
