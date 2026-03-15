# WITNESS ALPHA: Resource Cost vs Value Audit
**Lens: Every system evaluated by COST vs VALUE toward inevitability surfacing**
**Date: 2026-03-14**

---

## Methodology

I read every major per-step hook, all 14 agent mixin lifecycle methods, the EventBus, ConnectionRegistry, and bio-system wiring code. I traced which code runs every tick vs. every N steps vs. event-only. For the 12-agent daemon, one "step" = 12 agent.step() calls + 2 step hooks (_market_sense_hook and _sensing_step_with_advisory).

**Scale factor to keep in mind:** At pace=2.0, MIDGE runs ~172,800 steps/day. Every per-step cost is multiplied by that number.

---

## SYSTEM-BY-SYSTEM FINDINGS

---

### ConvergenceAlerter.check_convergence()
- **Category:** ESSENTIAL
- **Per-step cost:** Called every single step via `_market_sense_hook`. Iterates over in-memory `signals` dict (grouped by domain/direction), computes Thompson-weighted geometric mean, checks min_domains threshold, builds alerts. Also calls `check_ticker_convergence()` every step. Total: 2 full scans of the signal buffer per step.
- **Memory footprint:** Signal buffer in `data/market/signal_buffer.json` (29K signals per MEMORY.md). All loaded in-memory as defaultdict of lists. Each signal is a dataclass with ~10 fields.
- **Boot cost:** Loads signal_buffer.json, lag_correlations.json, registered_signals.json, Thompson distributions. Several hundred KB.
- **Value produced:** THE primary output. Multi-domain convergence detection is MIDGE's core function. Every alert is an inevitability candidate.
- **Cost/Value ratio:** Cost is justified. This is the engine.
- **Reasoning:** Cannot remove. However: calling `check_ticker_convergence(min_domains=3)` every step AND again at step%50 for Kelly sizing means the signal buffer is scanned 3 times per step. The Kelly scan at step%50 with min_domains=2 is redundant if done every step already.

---

### HypothesisEngine.step()
- **Category:** USEFUL
- **Per-step cost:** Called every step. Internally manages its own cadence (generation every 500 steps, validation every 1000 steps, regime check every 100 steps). The every-step overhead is the cadence check — three integer modulo operations + `getattr` calls. Actual work is cadence-gated.
- **Memory footprint:** HypothesisRegistry (JSONL-backed), in-memory hypothesis objects with probation/active/hibernated/retired states. ThreadPoolExecutor for async validation.
- **Boot cost:** Loads hypotheses.jsonl, retirement_window.json, seeds from lag findings.
- **Value produced:** RSI Layer 2 — converts lag/Granger findings into testable hypotheses. Active hypotheses fire `CH_HYPOTHESIS_FIRED` which boosts Focused Attention. Validated hypotheses feed Thompson sampler.
- **Cost/Value ratio:** Near zero per-step overhead (just modulo checks). Actual work is infrequent. Justified.
- **Reasoning:** The every-step call is architectural boilerplate — the real cost is only every 100/500/1000 steps. Appropriate.

---

### Thompson Sampler stats publish (every 10 steps)
- **Category:** ESSENTIAL
- **Per-step cost:** Every 10 steps: calls `sampler.get_stats(regime)` which iterates all 83 distributions + publishes to `CH_THOMPSON_STATS`. Every 75 steps: `regime_aware_forget(regime)` which iterates 83 distributions and decays.
- **Memory footprint:** 83 Beta distribution objects in RAM. ~5KB.
- **Boot cost:** Loads thompson_distributions.json. Runs `replay_from_history()` if distributions are empty.
- **Value produced:** The learning backbone. Without Thompson weights, convergence confidence is unweighted. Every-10-step publish keeps advisory consumers (bio systems, agents) current.
- **Cost/Value ratio:** Appropriate. 83 distributions is tiny. The every-10-step publish overhead is near zero.
- **Reasoning:** Essential to the learning loop. The 75-step forgetting cadence is gated behind outcome count change — won't fire if no outcomes have been graded, which is the correct behavior.

---

### MarketSensingHook._sensing_step_with_advisory()
- **Category:** ESSENTIAL
- **Per-step cost:** Wraps `hook.step()` (the actual async API fetch dispatch), reads `ctx._cached_alerts`, updates `_market_advisory`, calls `_run_paper_trading_gate()` (if alerts exist), every-10-step: queries 3 tiered alerters + runs `_run_sensing_archaeology()` (iterates ALL convergence signals to build active map, calls PatternWatcher.check()), every-20-step: `_run_active_tracker_check()`. EVERY step: `_run_synergy_detection()` (cross-checks convergence alerts vs pattern stacks).
- **Memory footprint:** Tiered alerters (3 separate ConvergenceAlerter instances with different windows). PatternWatcher template library (39 templates in RAM). ActiveTracker (up to 20 assets). Paper trade dedup dict.
- **Boot cost:** Loads paper_trades.jsonl for dedup state.
- **Value produced:** This is the data intake — 31 sources, 12 concurrent fetchers. Without this nothing enters the convergence engine.
- **Cost/Value ratio:** High cost, high value. Cannot remove. But `_run_synergy_detection()` runs EVERY step (checking convergence alerts vs pattern stacks for every ticker pair). This should be cadenced at 10 steps, matching archaeology.

---

### PatternWatcher.check() (every 10 steps, inside _run_sensing_archaeology)
- **Category:** ESSENTIAL
- **Per-step cost:** Every 10 steps. Builds active signal map from ALL convergence signals (iterates entire signal buffer), calls template matching against 39 templates, checks domain independence, builds PatternStack objects.
- **Memory footprint:** 39 templates in RAM (fingerprint IDs offloaded — only 3.3MB vs 100MB+). Stack cache.
- **Boot cost:** Loads pattern_library.jsonl. Validates templates. Zero fingerprints in RAM.
- **Value produced:** Pattern Archaeology is RSI Layer 4 — cross-symbol, cross-time template matching. When a stack fires alongside convergence, combined confidence compounds. The dual confirmation path is a genuine edge.
- **Cost/Value ratio:** Every-10-step scan of the signal buffer is the cost. Justified — but only because fingerprints are already offloaded. Before that fix, this would have consumed 100MB+ RAM.
- **Reasoning:** Essential but the every-step synergy detection check that reads the cache is unnecessary overhead.

---

### OctopusColony coordination (every 20 steps)
- **Category:** USEFUL
- **Per-step cost:** Every 20 steps: iterates all octopuses (up to N), runs `cognition.run_coordination_cycle()` for each, then iterates `_developing_situations` dict (up to 200 entries) and submits `investigate_partial` tasks (budget: 5/cycle). Each task submission is a thread-safe queue operation.
- **Memory footprint:** `_developing_situations` dict (up to 200 entries capped). Each OctopusAgent holds arm states. ThreadPoolExecutor for task workers.
- **Boot cost:** Bootstraps OctopusColony, OctopusAgents with cognition modules. Loads developing_situations.json.
- **Value produced:** Investigation pipeline — when partial convergences build up (2 domains, not 3), Octopus agents query PatternLibrary and WorldModel to find the missing domain evidence. This is the "Focused Attention" mechanism for developing situations.
- **Cost/Value ratio:** Moderate. The coordination cycle every 20 steps is lightweight. The value depends entirely on whether Octopus investigations actually resolve developing situations into full convergences. Without measurement of this completion rate, the cost is speculative.
- **Reasoning:** Architecturally sound but the actual ROI on investigation tasks completing into convergence alerts is unmeasured. This is the one market intelligence system whose value-production chain has the most unknowns.

---

### DeepAnalyst.analyze() (every 200 steps, inside _run_slow_cadence_ops)
- **Category:** ESSENTIAL
- **Per-step cost:** Every 200 steps. Synthesizes ranked inevitabilities from all data sources. Produces top-N `Inevitability` objects. Publishes to SituationBoard, persists to JSONL, embeds top-5 in Qdrant, formats plain-language alerts, registers for outcome grading.
- **Memory footprint:** Holds references to all market subsystems (convergence alerter, pattern library, world model, thompson sampler). Produces temporary list of inevitabilities.
- **Boot cost:** Minimal — stateless analyzer.
- **Value produced:** The synthesized inevitability ranking is the closest thing to MIDGE's "final answer" — the ranked list of what's structurally inevitable right now, combining convergence + pattern archaeology + world model + Thompson weights.
- **Cost/Value ratio:** High value, moderate cost (every-200-step cadence is reasonable). The Qdrant embedding on every run is potentially slow (5 network calls per analysis). Should be async.
- **Reasoning:** This is the crown jewel output layer. Essential.

---

### Lag Correlation Analyzer + Granger Analyzer (every 200 steps)
- **Category:** USEFUL
- **Per-step cost:** Every 200 steps each. Lag: reads signal archive (180-day lookback, 901+ files), computes cross-correlation with lag offsets. Granger: statsmodels bivariate F-test on domain pairs with Bonferroni correction.
- **Memory footprint:** Temporary arrays for the lookback window. Results stored in lag_correlations.json.
- **Boot cost:** None — stateless, on-demand analysis.
- **Value produced:** Lag findings feed directly into ConvergenceAlerter's sequence scoring (domain_sequence + sequence_score). Granger findings trigger HypothesisGenerator. These are the causal discovery engines — without them, convergence scoring doesn't have the temporal dimension.
- **Cost/Value ratio:** High value at low amortized cost (every 200 steps is fine for 90-day lookbacks). But reading 901 JSONL files every 200 steps at high step rates could cause I/O spikes.
- **Reasoning:** Keep. But consider: if steps run at pace=2.0, 200 steps = ~116 real seconds. Reading the archive every 2 minutes is aggressive. Could be extended to every 500 steps with minimal loss.

---

### Post-Mortem Reviewer (every 200 steps)
- **Category:** USEFUL
- **Per-step cost:** Every 200 steps. Reads outcomes.jsonl, analyzes combo stats, domain ordering, timing accuracy, MFE/MAE patterns, pushes Thompson updates.
- **Memory footprint:** Temporary dataframes for outcome analysis. Produces `post_mortem_insights.json`.
- **Boot cost:** None — stateless.
- **Value produced:** Closes the feedback loop on WHY predictions succeed/fail. Sequence-aware Thompson updates improve the learning engine beyond what standard outcome registration does.
- **Cost/Value ratio:** Good. Low amortized cost. Value is the timing intelligence (right thesis, wrong timing) which the standard Thompson loop doesn't capture.
- **Reasoning:** Keep.

---

### CascadeTracker + WorldModel Causal Watch
- **Category:** USEFUL
- **Per-step cost:** Event-driven, not per-step. `_on_signal_causal_watch` fires on every `CH_SIGNAL_INGESTED` event. `_on_signal_cascade_check` fires on every `CH_SIGNAL_INGESTED`. Both do dict lookups and BFS on the WorldModel graph (114 nodes, 102 edges — small).
- **Memory footprint:** WorldModel graph in RAM (~50KB). CascadeTracker active chains dict (bounded, expires stale). `_priority_requests` dict (capped at 50).
- **Boot cost:** WorldModel loads 114 nodes, 102 edges from hardcoded causal chains + discovers edges from lag/Granger at runtime.
- **Value produced:** Forward cascade: signals detected in one WorldModel node proactively watch for downstream effects. Backward cascade: mid-pattern discovery traces genesis. Sequential chain boost: confirmed cascades inject synthetic domain signals. These are all real value toward inevitability detection.
- **Cost/Value ratio:** Low cost (BFS on 114 nodes is ~microseconds), genuine value. The backward cascade discovery is particularly valuable — finding that a current signal is a domino in an existing cascade.
- **Reasoning:** Keep. The expire_stale every 200 steps is cheap and necessary for correctness.

---

### MycelialAgent.step() — 14 mixin lifecycle (12 agents per model step)
- **Category:** MIXED — parts REMOVABLE, parts USEFUL
- **Per-step cost:** 12 agents × 14 methods per step:
  - `_predict()`: requires `world_model` injection — in MIDGE, `world_model_enabled=False` for most agents (config default). Falls through to null path in 2 lines. Near-zero cost when disabled.
  - `_attend()`: requires `_attentional_gate` injection. If not injected, returns immediately. Near-zero.
  - `_observe()`: ALWAYS runs. Decays working memory, senses stigmergy markers (if env injected), builds state vector (numpy array construction). Reads body state from OrganismState. Reads pattern advisory ref. Checks theory_of_mind. Checks causal_engine. Checks predictive_field. **Most of these are MIDGE-irrelevant navigation/simulation concepts doing null work via getattr.**
  - `_compare()`: If `_last_prediction` is None (world model disabled), returns in 2 lines.
  - `_inhibit()`: ALWAYS runs InhibitionSystem.evaluate() — builds 8-input struct, calls evaluate() which computes a weighted score. Always runs regardless of market context.
  - `_decide()`: Always calls `organism.get_reflex_override()` which now returns None (pinned disabled). Chains through advisory router if available.
  - `_act()`: Executes selected action. For market agents pinned to api_call_enabled=False (Oracle shutdown), this does minimal work.
  - `_learn()`: Calls world model training, updates episodic memory, MAML updates. Most are gated behind None checks for injected subsystems.
  - `_manage_goals()`: GoalManager update.
  - `_communicate()`: Signal processing + GNN message processing + stigmergy deposit.
  - `_broadcast()` (every 3 steps): Global Workspace competitive ignition.
  - `_regulate()` (every 21 steps): Arousal homeostasis.
- **Memory footprint:** Per agent: ~20 attributes from 14 mixins. With 12 agents: ~240 total attribute bindings. State vectors (numpy arrays), episodic memory buffers, signal queues.
- **Boot cost:** Creating 12 full MycelialAgent instances with all 14 mixins initialized.
- **Value produced toward inevitability surfacing:** Near zero. The agents' _decide/_act loop (selecting explore/exploit/communicate/rest actions) produces no market signals. Their market contribution is: (1) consuming advisory state from ctx, (2) depositing stigmergy markers, (3) running MAML/transfer learning updates (if enabled).
- **Cost/Value ratio:** 12 agents × ~20 method calls/step × 172,800 steps/day = ~41 million method calls/day executing primarily null paths or bio-simulation work that has been pinned/disabled. The legitimate work (InhibitionSystem, Advisory routing) runs 12 times per step when it only needs to run once.
- **Reasoning:** The agents are bio-simulation infrastructure carrying market advisory state. Their per-step cost is dominated by null-path getattr chains. Particular waste areas: `_observe()` checking theory_of_mind, causal_engine, predictive_field (all null in MIDGE), `_broadcast()` running GWT ignition 12 times per step, `_regulate()` running arousal homeostasis 12×/21-step cycle.

---

### EndocrineSystem.step()
- **Category:** USEFUL (low cost)
- **Per-step cost:** Registered as a step hook (from bootstrap). Decays all hormone levels toward baseline (8 hormone types, 8 float operations). Publishes `CH_HORMONE_STATE` every N steps (configurable interval).
- **Memory footprint:** 8 float hormone levels + config dict. Negligible.
- **Boot cost:** Minimal.
- **Value produced:** Provides global arousal state that modulates agent behavior. Wired to market events: convergence → dopamine/adrenaline, deception → cortisol. The market coupling is real — convergence alerts trigger dopamine which feeds through to agent reward signals.
- **Cost/Value ratio:** Very low cost, moderate value. The endocrine system is a lightweight global state modulator.
- **Reasoning:** Keep. 8 float decays per step is negligible overhead.

---

### CircadianRhythm.step()
- **Category:** INERT (with one HARMFUL behavior pinned)
- **Per-step cost:** Registered as a step hook. Advances phase counter, occasionally fires `CH_PHASE_CHANGE`. The phase multiplier that would cut sensing workers to 25% has been PINNED: `ctx._circadian_activity = 1.0` in bio_market_wiring_b.py.
- **Memory footprint:** Phase state, counter. Negligible.
- **Boot cost:** None.
- **Value produced:** MemoryConsolidator uses phase changes to trigger hypothesis consolidation (CONSOLIDATION phase) and excavation (REST phase). This is the only real market job.
- **Cost/Value ratio:** Very low cost. The value is the phase-triggered consolidation, which is a legitimate cadence mechanism — but it duplicates the explicit step-counter cadencing already in `_market_sense_hook`. The MemoryConsolidator calling `hypothesis_engine.step()` on CONSOLIDATION is a second invocation of something already called every step.
- **Reasoning:** Near-inert. The pinning of sensing workers at 1.0 is correct. The MemoryConsolidator callback creates a duplicate hypothesis_engine.step() invocation. Minor issue.

---

### OrganismState (18-subsystem body state aggregator)
- **Category:** USEFUL (partially REMOVABLE)
- **Per-step cost:** Event-driven callbacks from 18 bio systems update float attributes. `get_reflex_override()` is called every agent step (12×/step) — now returns None immediately (pinned). `get_body_state()` is called every agent step to populate `_body_state`. `report_action_outcome()` runs on every agent action.
- **Memory footprint:** ~25 float attributes, a deque of recent outcomes. Negligible.
- **Boot cost:** Subscribes to ~30 EventBus channels.
- **Value produced:** The `get_decision_context()` output (body_threat_level, body_opportunity_level, emotional_bias, metacognitive_confidence, organism_vitality) feeds into advisory routing. The vitality EMA is a real signal — if predictions are consistently winning, vitality rises, which propagates into the organism state accessible to agents.
- **Cost/Value ratio:** Low cost. The market-meaningful outputs are `emotional_bias` (driven by market convergence events) and `vitality` (driven by prediction outcomes). The remaining 18 bio-system attributes (oxygen_level, toxin_load, circulation_adequate, etc.) are populated but their consumer (get_reflex_override) is permanently disabled.
- **Reasoning:** Keep the vitality + emotional state machinery. The 30 channel subscriptions for bio-system channels (lymph_status, respiration_update, hypoxia, etc.) are overhead for attributes that nothing reads in a meaningful way now that reflex override is disabled.

---

### Bio Systems Tier 2 (EmotionalSystem, HomeostasisRegulator, ArousalRegulator, CuriosityDrive, NociceptionSystem)
- **Category:** USEFUL (low cost, low value)
- **Per-step cost:** Event-driven only — callbacks fire on CH_CONVERGENCE, CH_DECEPTION_DETECTED, CH_PREDICTION_RESULT, CH_VELOCITY_ANOMALY. No per-step polling.
- **Memory footprint:** Float attributes per system. Negligible.
- **Boot cost:** Register 2-3 callbacks each.
- **Value produced:**
  - EmotionalSystem: Updates `_surprise_boost`/`_fear_reinforcement` floats → feeds OrganismState emotional state → feeds `get_decision_context()` → feeds advisory routing. The chain exists but the emotional state's actual influence on final trades is thin.
  - HomeostasisRegulator: Updates threat_level/energy_level setpoints. OrganismState reads these but reflex override is disabled so it never triggers rest.
  - ArousalRegulator: Records reward from prediction outcomes. The Yerkes-Dodson optimal arousal concept has no downstream consumer that acts on it differently from raw reward.
  - CuriosityDrive: Adjusts `_exploration_bonus`. This bonus feeds into something if agents are in EXPLORE mode — but Oracle agents are api_call_enabled=False.
  - NociceptionSystem: Reports damage as pain. OrganismState accumulates `_pain_load` but reflex override that would act on pain is disabled.
- **Cost/Value ratio:** Near zero cost. Near zero value. The callbacks run on events, they update floats, those floats mostly update OrganismState attributes that are never acted on (because reflex is disabled).
- **Reasoning:** Inert. They run without harm but produce no market intelligence. They are decorative bio-simulation. If reflex override were re-enabled with market-meaningful thresholds, they would have value.

---

### Bio Systems Tier 3 (MetacognitionMonitor, ThreatDetector, HAVEN, InhibitionSystem)
- **Category:** MIXED
- **Per-step cost:** Event-driven callbacks only.
- **Memory footprint:** Per-system state. Negligible individually.
- **Value produced:**
  - MetacognitionMonitor: Records prediction confidence vs actual outcome. Feeds learning rate adjustment. This is a real feedback signal — if confidence is systematically mis-calibrated, the metacognition monitor could flag it. But its output (`metacognition_score` in OrganismState) isn't demonstrably influencing paper trade thresholds.
  - ThreatDetector: Registers deception quills, registers sacrificeable components. The sacrificeable component list includes `finnhub_websocket`, `apewisdom_client`, `fractal_resonance_detector`, `pattern_completion_engine`. This is real — under threat, MIDGE could drop low-priority components. But the mechanism (ThreatDetector scanning quills, autotomy) has never actually fired.
  - HAVEN: Tracks `ctx._haven_market_flags` per source. Increments on deception, decrements on success. But these flags are not read anywhere in the signal processing pipeline to actually discount signals from flagged sources.
  - InhibitionSystem: Sets `ctx._market_caution`. This IS read in `_run_paper_trading_gate()` and applies up to 30% confidence penalty on paper trades. This is the only Tier 3 bio system with a real, measured effect on trading decisions.
- **Cost/Value ratio:**
  - MetacognitionMonitor: Low cost, unmeasured value.
  - ThreatDetector: Low cost, dormant value (mechanism never fired).
  - HAVEN: Low cost, ZERO value (flags exist but nothing reads them in the signal pipeline).
  - InhibitionSystem: Low cost, REAL value (market_caution actively penalizes paper trades).
- **Reasoning:** InhibitionSystem: KEEP. HAVEN: REMOVABLE unless the flags are wired into signal source weighting. MetacognitionMonitor: borderline — keep for observability. ThreatDetector: keep for future use (component sacrifice under threat is a valid mechanism).

---

### Bio Systems Tier 3 (CircadianRhythm, MemoryConsolidator, CollectiveDreamPlanner, QuorumSpace, Stigmergy)
- **Category:** INERT to REMOVABLE
- **Per-step cost:** Event-driven.
- **Value produced:**
  - CircadianRhythm: Already covered above. Mostly inert, one real job (phase-triggered consolidation, duplicate).
  - MemoryConsolidator: Triggers hypothesis_engine.step() on CONSOLIDATION phase and excavation_daemon.step() on REST phase. These are legitimate jobs but duplicates of explicit cadencing.
  - CollectiveDreamPlanner: On convergence, increments `expertise` for each dreamer by 0.02. The CollectiveDream runs... dream planning sessions for agents. In MIDGE, agents don't execute a dream planner-driven behavior that produces market signals. This is decorative.
  - QuorumSpace: Deposits convergence/pattern stack/dual confirmation signals into a quorum bucket. Tracks which ticker+direction has quorum. But the quorum result is never read to gate or boost any market decision in the pipeline. The data accumulates, nothing consumes it.
  - Stigmergy: Deposits pheromone markers at ticker positions when convergences fire and when predictions resolve. Agents read stigmergy markers in `_observe()` as gradients. They follow SUCCESS trails and avoid DANGER trails. But agents are Oracle-shutdown (api_call_enabled=False) so even if an agent is attracted toward a high-convergence ticker, the "action" they take is `rest` or `communicate` — not a market action.
- **Cost/Value ratio:** Low cost, near-zero market value for all five.
- **Reasoning:** CollectiveDreamPlanner is REMOVABLE from bio-market wiring (the dream itself has no market downstream). QuorumSpace is REMOVABLE unless its output is wired to a decision gate. Stigmergy deposits are REMOVABLE as long as agents are Oracle-shutdown (the trails have no consumers that produce market signals).

---

### Bio Systems Tier 4+5 (DigestiveSystem, CirculatorySystem, LymphaticSystem, Microbiome, RenalFilter, SenescenceManager, MorphogenesisCoordinator, ReproductiveSystem, PearlDefense, RespiratorySystem, ThermoregulationSystem, VestibularSystem, ProprioceptionSystem, EnergyReserve, PredictiveField)
- **Category:** INERT to REMOVABLE (with two partial exceptions)
- **Per-step cost:** Event-driven callbacks only. Some have thread locks (ReproductiveSystem uses `_pressure_lock`, MorphogenesisCoordinator uses `_partial_lock`).
- **Memory footprint:** Per-system state, mostly float attributes. MorphogenesisCoordinator maintains `_partial_window` deque (maxlen=100).
- **Value produced:**
  - DigestiveSystem: `ingest()` called on convergence/partial. The digestion queue has no downstream consumer that gates convergence processing.
  - CirculatorySystem: `request_resource()` called on convergence/velocity. Tracks "heart rate" rising under load. OrganismState reads circulation via `_circulation_adequate`. With reflex override disabled, this never affects agent behavior.
  - LymphaticSystem: Collects "waste" (failed predictions). The waste collection has no downstream consumer that cleans up stale signals from the convergence buffer.
  - Microbiome: `process_input()` called on convergence/velocity/pattern stacks. Tracks microbial diversity. OrganismState reads microbiome_diversity but reflex is disabled.
  - RenalFilter: `add_toxin_pattern()` and `filter_item()` called on deception events. Calls `time.monotonic_ns()` for item IDs (minor overhead). The filter result verdict is logged but not acted on.
  - SenescenceManager: Reports activity on convergence/prediction. Tracks wear on market systems. If a system stops firing, it accumulates wear. Wear could trigger rejuvenation. But there is no wiring to actually re-bootstrap a worn-out market system.
  - MorphogenesisCoordinator: Tracks partial convergence rate, calls `morph.handle_novel_problem()` when rate >= 5/10min. Morphogenesis spawns new investigation organs — but the OctopusColony already handles investigation. Duplicate mechanism.
  - ReproductiveSystem: Tracks `_market_activity_pressure`. `consume_market_pressure()` is stored on ctx but is it ever called? Not found in the step hooks examined. The pressure accumulates but is never consumed.
  - PearlDefense: Validates deception signals with nacre process. No downstream consumer reads the validation result to gate signal acceptance.
  - RespiratorySystem: Explicitly disabled (oxygen drain removed, callbacks not registered). Returns 1 for wiring count.
  - ThermoregulationSystem: `report_activity()` called on convergence/velocity. OrganismState reads `_temperature_zone`. No actionable consequence with reflex disabled.
  - VestibularSystem: `report_metric()` called on convergence/prediction. OrganismState reads `_stability`. No actionable consequence with reflex disabled.
  - ProprioceptionSystem: `update_position()` called on convergence/prediction for convergence_alerter and outcome_tracker. Body map updated but never read in a way that changes behavior.
  - EnergyReserve: `store(5.0)` on REST phase, `release(1.0)` on ACTIVE phase. `is_critical()` pinned to False. `step()` publishes `CH_ENERGY_STATUS` every step — this fires into OrganismState `_on_energy_status` callback every step. The EnergyReserve.step() method publishes on EVERY call (no cadence gating): CH_ENERGY_STATUS, CH_STARVATION (if critical, pinned False), CH_RESERVES_FULL (if full). At 100.0/200.0 capacity it fires CH_RESERVES_FULL every step it is invoked. That's a constant stream of EventBus publishes → JSON serialization → callback dispatch to OrganismState → float updates, accomplishing nothing.
  - PredictiveField: `update_agent_state()` called on convergence. Field is read in `_observe()` for collision risks and coordination opportunities — but the collision avoidance block is DISABLED (dead code in lifecycle_inhibit_decide.py).
- **Cost/Value ratio:** Collectively: low individual costs that compound. EnergyReserve.step() is a standout: publishing 2-3 EventBus messages every step it runs (including CH_RESERVES_FULL when at 100% — which is always since drain is removed) means constant JSON serialization overhead for zero market value.
- **Reasoning:** The entire Tier 4+5 bio system wiring produces no market intelligence and no actionable decisions (all reflex paths are disabled). They are running metaphor — computation that correctly describes biological analogues but produces no signal that reaches a convergence check, Thompson update, or trading decision.

---

### EventBus
- **Category:** ESSENTIAL (with overhead concern)
- **Per-step cost:** Each `publish()` call acquires a threading.RLock, JSON-serializes the message (using NumpySafeEncoder), checks ConnectionRegistry for triadic witnessing on every non-system channel, then iterates all callback subscribers. With bio-market wiring, many channels have 5-15 callbacks (CH_CONVERGENCE alone triggers: EmotionalSystem, HomeostasisRegulator, ArousalRegulator, InhibitionSystem, Stigmergy, CirculatorySystem, DigestiveSystem, RenalFilter, QuorumSpace, ThermoregulationSystem, VestibularSystem, ProprioceptionSystem, PredictiveField, CollectiveDreamPlanner + more = ~15 callbacks). Every convergence alert publish triggers 15+ JSON deserialization + float update operations.
- **Memory footprint:** `_subscribers` dict (callback lists per channel), `_streams` defaultdict (deque(maxlen=10000) per stream, holds dict entries). Stream storage could be significant if many channels use stream API.
- **Boot cost:** Negligible.
- **Value produced:** Essential infrastructure. All market intelligence flows through it.
- **Cost/Value ratio:** The bus itself is essential. The overhead comes from the accumulated bio-system callbacks on high-frequency channels. CH_CONVERGENCE is the most expensive channel: each convergence alert publication triggers ~15 callbacks, each doing JSON deserialization + attribute update, most of which produce no market value.
- **Reasoning:** The EventBus is not the problem — the problem is the subscriber density on CH_CONVERGENCE from inert bio systems. Removing 10 inert bio callbacks from CH_CONVERGENCE would reduce its per-publish overhead by ~67%.

---

### ConnectionRegistry.is_connection_allowed() on every publish()
- **Category:** HARMFUL (unnecessary overhead)
- **Per-step cost:** Every EventBus publish() call (when sealed) calls `is_connection_allowed(source, "event_bus", channel=channel)`. This runs a dict lookup + advisory check. The source is extracted with `channel.split(".")[0]` for every single message. With ~hundreds of publishes per step (bio-system status updates, Thompson stats, convergence alerts, cascade events, etc.), this adds up.
- **Memory footprint:** ConnectionRegistry holds 428 connections. Lookup is O(1) dict.
- **Boot cost:** Seals the registry after bootstrap.
- **Value produced:** Advisory enforcement of triadic witnessing. In ADVISORY mode (the default), it never actually blocks anything — it just logs. So it's running a check that produces no enforcement effect.
- **Cost/Value ratio:** Every publish() → split() + dict lookup + advisory check, for a system that only logs and never blocks. In advisory mode, this is pure overhead.
- **Reasoning:** The check could be removed from the hot path and moved to a periodic audit (every 1000 steps, check all channel registrations). Alternatively, pre-compute "unchecked" channels and skip the lookup for known-safe system channels.

---

### HolonRegistry.register() / HolonMixin per-step
- **Category:** INERT (per-step cost near zero)
- **Per-step cost:** `holon_know_self()` is called every `_observe()` for every agent (12 calls/step). This calls `_holon_registry.get_holon(holon_id)` which is an O(1) dict lookup returning the agent's own registered state. Then it reports awareness pulse. Low cost.
- **Memory footprint:** 157 holons × ~8 fields each. A few KB.
- **Boot cost:** Registers all 157 holons with their parent/child relationships. Runs awareness pulse initialization.
- **Value produced:** The holon protocol is the architectural guarantee that every entity has 10 capabilities. For agents, `know_self()` returns their current capability state. This is used in the advisory routing — agents with high self-assessed capability for a domain can be preferentially routed tasks. But in practice, agents in Oracle shutdown mode don't execute domain-specific tasks.
- **Cost/Value ratio:** Low cost, low market value.
- **Reasoning:** Architecturally mandated (Law 3). The per-step cost is acceptable. The value toward inevitability surfacing is indirect (agent routing).

---

### StigmergyEnvironment (evaporation, every 50 steps)
- **Category:** INERT
- **Per-step cost:** Every 50 steps: `sense_markers()` with radius=infinity is called to trigger evaporation of all markers. This is an O(N) scan over all deposited markers.
- **Memory footprint:** The marker store holds all pheromone deposits. With convergence alerts firing constantly, the marker store could be large (every convergence deposits 1-2 markers).
- **Boot cost:** Creates the spatial marker storage.
- **Value produced:** Agents follow SUCCESS/DANGER trails in `_observe()`. But as noted: agents are Oracle-shutdown, so trail-following doesn't produce market actions.
- **Cost/Value ratio:** O(N) evaporation scan every 50 steps, for a trail system whose consumers are disabled. The marker store grows unbounded between evaporation cycles.
- **Reasoning:** REMOVABLE from market step hooks until agents have market-executable actions. The bio-market wiring that deposits markers on CH_CONVERGENCE (every convergence event) should be suspended until Stigmergy consumers are producing market signals.

---

### FinnhubWebSocket (background thread)
- **Category:** ESSENTIAL
- **Per-step cost:** Background thread — no per-step overhead in main loop. Pushes realtime prices to signal buffer when tickers have active monitoring. Rate of pushes depends on subscription count.
- **Memory footprint:** WebSocket connection state, message queue.
- **Boot cost:** Starts background thread, establishes WebSocket connection.
- **Value produced:** Real-time price streaming for active tickers. Provides `finnhub_realtime` signals to convergence engine. The only truly real-time data source (all others are polled).
- **Cost/Value ratio:** Low step overhead, high signal value for active market hours.
- **Reasoning:** Keep.

---

### ThompsonCalibrator (every 500 steps)
- **Category:** USEFUL
- **Per-step cost:** Every 500 steps. Calibrates Thompson distributions against recent outcomes. Adjusts distribution parameters to match observed win rates.
- **Memory footprint:** Reads from outcomes.jsonl. Temporary computation.
- **Boot cost:** None.
- **Value produced:** Corrects systematic bias in Thompson distributions. Without calibration, distributions can drift from actual win rates. Essential to keeping confidence scores meaningful.
- **Cost/Value ratio:** Low amortized cost, direct value to the learning engine.
- **Reasoning:** Keep.

---

### RegimeClassifier
- **Category:** ESSENTIAL
- **Per-step cost:** `_get_regime()` is called every step in `_market_sense_hook`. The classify() method reads recent SPY prices and computes volatility/trend. Result is cached (classified as "daily") so repeated calls return cached value cheaply.
- **Memory footprint:** Small price history buffer.
- **Boot cost:** Minimal.
- **Value produced:** Regime (bull/bear/volatile/sideways) controls Thompson decay rates (`REGIME_DECAY_RATES`). Different regimes = different forgetting speeds. This is a real differentiation — volatile regimes decay Thompson faster (correct, since signals change meaning faster).
- **Cost/Value ratio:** Low cost, direct value to the learning engine.
- **Reasoning:** Essential.

---

### DrawdownMonitor + SystemHealthMonitor
- **Category:** ESSENTIAL
- **Per-step cost:** DrawdownMonitor tracks position P&L and daily drawdown. SystemHealthMonitor tracks success/error rates for 8 subsystems. Both are lightweight counter-based. DrawdownMonitor calls `is_trading_halted()` before every paper trade.
- **Memory footprint:** Counters + recent trade records. Negligible.
- **Boot cost:** None.
- **Value produced:** Risk management circuit breakers. Without drawdown monitoring, MIDGE could blow through daily loss limits on an FTMO challenge. Direct financial value.
- **Cost/Value ratio:** Near-zero cost, essential risk value.
- **Reasoning:** Essential.

---

### ExcavationDaemon (every 2000 steps, off main thread)
- **Category:** USEFUL
- **Per-step cost:** Every 2000 steps: submits a background task to the sensing hook's ThreadPoolExecutor. Actual excavation (fetching historical data, building fingerprints, accumulating templates) runs off-thread.
- **Memory footprint:** 274K fingerprints offloaded to disk (only 3.3MB fingerprint_ids set in RAM). 39-43 templates in RAM.
- **Boot cost:** Loads PatternLibrary (templates only). Fingerprints are ID-indexed.
- **Value produced:** Continuously extends the archaeological record. New fingerprints improve template coverage. Cross-symbol validation of templates requires fingerprint accumulation.
- **Cost/Value ratio:** Minimal main-thread cost (off-thread work). Excavation is complete (3,205/3,237 symbols per MEMORY.md). The marginal value of continued excavation is diminishing.
- **Reasoning:** Keep but reconsider cadence. With excavation 99% complete, the daemon's remaining value is re-checking already-excavated symbols for new patterns as more historical data accumulates. Every-2000-step cadence is appropriate.

---

### SensingHook — 12 concurrent API fetchers
- **Category:** ESSENTIAL
- **Per-step cost:** Dispatches async fetches across 31 sources using ThreadPoolExecutor (12 workers). Cadence = 25 steps per source rotation (approximately). Each fetch is I/O-bound (network calls) so CPU overhead on main thread is just task submission.
- **Memory footprint:** Signal buffer holds all fetched signals. 29K signals per MEMORY.md at ~200 bytes/signal = ~6MB.
- **Boot cost:** Creates ThreadPoolExecutor(12). Initializes all 31 client instances.
- **Value produced:** ALL market data intake. Without fetchers, nothing enters the convergence engine. Every signal that becomes a convergence alert starts here.
- **Cost/Value ratio:** Essential. The 12-worker thread pool is justified by the 31 sources and the need for parallel fetching.
- **Reasoning:** Keep.

---

### GNN Communicator (if injected)
- **Category:** REMOVABLE (in MIDGE context)
- **Per-step cost:** `process_gnn_messages()` called in `_communicate()` every agent step. Iterates pending GNN messages for each agent. In MIDGE, GNN routing optimizes inter-agent message routing — but agents in Oracle shutdown don't receive or act on market-specific tasks via GNN messages.
- **Memory footprint:** Per-agent message queues.
- **Boot cost:** Creates GNN model for message routing.
- **Value produced toward inevitability surfacing:** None identified. Agent-to-agent communication in MIDGE is not producing market intelligence.
- **Cost/Value ratio:** Processing GNN messages 12 times per step for zero market output.
- **Reasoning:** REMOVABLE in MIDGE's current operating mode. The GNN serves agent coordination in a general simulation context, not in a market daemon where agents are Oracle-shutdown.

---

### Transfer Learning + MAML (EpisodicMemoryMixin)
- **Category:** INERT
- **Per-step cost:** `_learn()` in lifecycle_learning.py calls MAML update, transfer learning, episodic memory storage every step (if respective subsystems are injected). In MIDGE, the "tasks" being learned are the generic agent reward signals (explore/exploit/communicate/rest), not market-specific tasks.
- **Memory footprint:** Episodic memory buffer per agent (configurable size). Transfer knowledge base. MAML model weights.
- **Boot cost:** Creates MAML learner, episodic memory store.
- **Value produced toward inevitability surfacing:** Near zero. The MAML/transfer learning subsystem is optimizing agents for the generic Mesa simulation action space, not for market intelligence tasks.
- **Cost/Value ratio:** Non-trivial cost (MAML involves gradient computation if enabled), zero market value.
- **Reasoning:** INERT. The MAML/transfer learning improvements don't transfer to market signal quality. The agents' "reward" is not market P&L; it's the generic simulation reward function.

---

### GlobalWorkspace._broadcast() (every 3 steps per agent)
- **Category:** REMOVABLE (in MIDGE context)
- **Per-step cost:** Every 3 steps, each of 12 agents runs competitive ignition in the Global Workspace (GWT). This selects a "winning" signal from the agent's recent signals and broadcasts it. In MIDGE, agents don't have market-specific signals — they have the generic explore/exploit/communicate/rest action signals.
- **Memory footprint:** Per-agent competition state, recent signal buffer.
- **Boot cost:** Creates GlobalWorkspace for each agent.
- **Value produced toward inevitability surfacing:** None identified. The GWT competition is broadcasting agent action decisions, not market insights.
- **Cost/Value ratio:** 4 agent broadcasts per step (12 agents / 3 cadence) with zero market signal output.
- **Reasoning:** REMOVABLE from market daemon context. GWT serves a deliberate attention mechanism in cognitive architectures; without market-specific "signals" entering the competition, it optimizes nothing relevant.

---

## SUMMARY TABLE

| System | Category | Per-Step Overhead | Market Value |
|--------|----------|-------------------|--------------|
| ConvergenceAlerter (every step) | ESSENTIAL | High | CORE |
| MarketSensingHook | ESSENTIAL | High | CORE |
| ThompsonSampler | ESSENTIAL | Low (every 10) | CORE |
| RegimeClassifier | ESSENTIAL | Low (cached) | CORE |
| HypothesisEngine | ESSENTIAL | Near zero (cadence-gated) | HIGH |
| DeepAnalyst | ESSENTIAL | Low (every 200) | HIGH |
| DrawdownMonitor + SHM | ESSENTIAL | Near zero | HIGH (risk) |
| FinnhubWebSocket | ESSENTIAL | Zero (background thread) | HIGH |
| CascadeTracker + WorldModel | USEFUL | Near zero (event-driven) | MEDIUM |
| PatternWatcher (every 10) | ESSENTIAL | Medium | HIGH |
| OctopusColony (every 20) | USEFUL | Low | MEDIUM (unmeasured ROI) |
| Lag/Granger analyzers (every 200) | USEFUL | Low (amortized) | MEDIUM |
| Post-Mortem Reviewer (every 200) | USEFUL | Low | MEDIUM |
| ThompsonCalibrator (every 500) | USEFUL | Near zero (amortized) | MEDIUM |
| ExcavationDaemon (every 2000, off-thread) | USEFUL | Near zero (amortized) | LOW-MEDIUM |
| EndocrineSystem | USEFUL | Near zero | LOW |
| InhibitionSystem (bio-market) | USEFUL | Near zero | LOW (paper trade gate) |
| OrganismState | USEFUL (partial) | Low (12 reads/step) | LOW |
| MetacognitionMonitor | INERT | Near zero | NEAR ZERO |
| ThreatDetector | INERT | Near zero | NEAR ZERO (potential) |
| HolonRegistry | INERT | Near zero | NEAR ZERO |
| CircadianRhythm | INERT | Near zero | NEAR ZERO |
| MemoryConsolidator | INERT | Near zero | NEAR ZERO (duplicate) |
| Tier 2 bio systems (5) | INERT | Near zero | NEAR ZERO |
| HAVEN | REMOVABLE | Near zero | ZERO (flags unread) |
| QuorumSpace | REMOVABLE | Near zero | ZERO (output unconsumed) |
| CollectiveDreamPlanner | REMOVABLE | Near zero | ZERO |
| Stigmergy (deposits) | REMOVABLE | Near zero | ZERO (consumers disabled) |
| Tier 4+5 bio systems (15) | REMOVABLE | Low (event-driven but constant) | ZERO |
| EnergyReserve.step() | HARMFUL | Medium (constant EventBus publishes) | ZERO |
| ConnectionRegistry on publish() hot path | HARMFUL | Low (per every publish) | ZERO (advisory-only) |
| MycelialAgent bio-simulation overhead | HARMFUL | High (12 agents × null paths) | ZERO |
| GNN Communicator (agent-to-agent) | REMOVABLE | Low | ZERO |
| Transfer Learning / MAML | INERT | Medium (if enabled) | ZERO |
| GlobalWorkspace._broadcast() | REMOVABLE | Low (12 agents every 3 steps) | ZERO |

---

## TOP RESOURCE DRAINS WITH NO MARKET VALUE

### Drain 1: Bio-market wiring on CH_CONVERGENCE (15+ callbacks, mostly inert)
Every convergence alert triggers ~15 EventBus callbacks: DigestiveSystem.ingest(), CirculatorySystem.request_resource(), RenalFilter.filter_item(), Microbiome.process_input(), SenescenceManager.report_activity(), StigmergyEnvironment.deposit_marker(), ThermoregulationSystem.report_activity(), VestibularSystem.report_metric(), ProprioceptionSystem.update_position(), PredictiveField.update_agent_state(), EmotionalSystem (fear/surprise update), HomeostasisRegulator (threat level), QuorumSpace.deposit_signal(), CollectiveDreamPlanner (expertise bump). Only 2-3 of these produce effects that reach paper trade decisions.

**Estimated waste:** Each convergence alert publication = ~12 wasted callback dispatch + JSON deserializations per alert. At busy sessions with many alerts, this creates a meaningful tax on alert processing.

### Drain 2: EnergyReserve publishing CH_RESERVES_FULL every step
EnergyReserve.step() runs its full publish logic every time it's called (if invoked). Reserves are pinned at 100.0 and drain is disabled, so `is_full()` = True on every step (100.0 > 200.0 * 0.9 = 180.0 — actually False, 100 < 180). Wait, 100/200 = 50% capacity. is_full() checks > 90%, so it's False. But CH_ENERGY_STATUS is published on EVERY step.call() — unconditionally (line 241: `if self._bus is not None: self._bus.publish(...)`). OrganismState._on_energy_status fires, updating float attributes that nothing acts on.

**Estimated waste:** One EventBus publish + JSON serialization + OrganismState float updates per EnergyReserve step call. If called at every circuit-breaker step, this is constant overhead.

### Drain 3: ConnectionRegistry triadic check on every EventBus publish
Every publish() call in advisory mode runs `is_connection_allowed()` which: extracts source from channel name (string split), looks up the connection registry, calls `is_connection_allowed()`. Never blocks anything. Pure overhead.

**Estimated waste:** O(1) overhead per publish, but with hundreds of publishes per step, this adds up to a meaningful constant.

### Drain 4: 12-agent lifecycle overhead for bio-simulation null paths
`_observe()` for each of 12 agents runs: theory_of_mind check (getattr + None check), causal_engine check (getattr + None check), predictive_field collision check (getattr + None check for both collision_risks and coordination_opps). All return None/empty. This is 12 × ~10 null-path getattr chains per step.

**Estimated waste:** Low individually, cumulative at 172K steps/day = ~20M null-path getattr chains/day that produce nothing.

---

## SPECIFIC ACTIONABLE FINDINGS

1. **_run_synergy_detection() runs every step** — should be cadenced with `_run_sensing_archaeology()` (every 10 steps). The synergy check reads cached data so every-step is redundant.

2. **CH_CONVERGENCE has ~15 subscribers**, ~12 of which produce no market value. Removing inert bio-market callbacks from CH_CONVERGENCE alone would cut its per-publish cost by ~80%.

3. **EnergyReserve.step() publishes CH_ENERGY_STATUS unconditionally.** If this is called every step (bootstrap wires it as a step hook — confirmed in bio-market wiring extended), this is a constant EventBus publish with no downstream market effect. Should be cadenced (e.g., every 100 steps) or the publish gated behind a change threshold.

4. **ReproductiveSystem's consume_market_pressure() is never called.** `ctx._consume_market_pressure` is stored but grep-tracing the step hooks shows no invocation. The pressure accumulates toward 1.0 and is never consumed. Dead accumulator.

5. **HAVEN's `ctx._haven_market_flags` dict grows unbounded.** It accumulates source flags on deception events and clears them on prediction successes for listed sources. But it is never read in the signal processing pipeline. The flags exist and grow.

6. **The 3x convergence buffer scan per step:** `check_convergence()` scans signals, `check_ticker_convergence(min_domains=3)` scans again, and at step%50 `check_ticker_convergence(min_domains=2)` scans a third time. The step%50 Kelly sizing check scanning with min_domains=2 is the most redundant — it would catch signals already caught by the every-step min_domains=3 check plus some weaker signals, but those weaker signals only matter for Kelly sizing, not for convergence alerts.

7. **OctopusColony investigation ROI is unmeasured.** The system submits up to 5 investigation tasks every 20 steps (potentially 60 task submissions per minute). Each task queries PatternLibrary and WorldModel for the developing situation. But there is no metric tracking how many investigation tasks result in a partial situation being resolved into a full convergence alert. Without this number, it's impossible to know if the investigation pipeline is delivering value or just generating noise.

---

*Alpha findings complete. Independent of Beta and Gamma analyses.*
