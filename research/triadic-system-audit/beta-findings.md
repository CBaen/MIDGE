# MIDGE System Audit — Beta Findings (Adversarial Lens)

**Auditor:** Witness Beta (Devil's Advocate)
**Date:** 2026-03-14
**Method:** Full codebase read — main.py, all bootstrap layers, all bio systems, all market systems, agent architecture, backbone, defense
**Scope:** What is MIDGE's purpose? "Observe patterns, find where converging forces make outcomes structurally inevitable, surface those for humans." Every system is judged against this.

**Grand total:** 649 Python files, 120,571 lines of code. Market module = 54,229 lines. Everything else (organism overhead) = 66,438 lines. The organism is bigger than the organ it supposedly exists to serve.

---

## The Headline Finding

MIDGE is a market intelligence data pipeline with a 33-layer organism simulator strapped to its back. The organism does not help the data pipeline. In most cases, the organism was found to HARM the data pipeline, and has been progressively disabled — leaving dead code, pinned values, commented-out callbacks, and `return None` stubs where real behavior used to be.

The code itself documents this: 14+ explicit `# MIDGE: disabled — fictional physiology harms trading daemon` comments exist across `lifecycle_inhibit_decide.py`, `organism_state_outputs.py`, `inhibition_system.py`, `energy_reserve.py`, `bio_market_wiring.py`, `bio_market_wiring_b.py`, `bio_market_wiring_extended_b.py`. This is not a minor issue. It is the codebase admitting that the organism is a liability to the market mission.

---

## The 33-Layer Bootstrap

### main.py (1,092 lines, 8 bootstrap phases)

- **Category:** OVERENGINEERED but structurally necessary
- **The honest question:** What would break if layers were reduced?
- **Evidence:** The 33 "layers" decompose into 8 actual bootstrap phases. Real work happens in 4: foundation (EventBus, basic systems), agents (MycelialAgent creation), wiring (ConnectionRegistry), and market (Layer 33 — the actual purpose). Layers 22-30 (patterns, organs, lifecycle) exist purely to satisfy the organism model.
- **Verdict:** The bootstrap could be collapsed to 5 meaningful phases without loss of market capability. The 33-layer count is a narrative artifact, not an architectural necessity. Keep the structure, but acknowledge that 25+ layers are overhead.

---

## Agents — The Central Question

### MycelialAgent (14 mixins, 12-step lifecycle)

- **Category:** HARMFUL (net negative for market intelligence)
- **The honest question:** What would break if MIDGE had no agents at all?
- **Evidence:**
  - `lifecycle_act.py` line 28: `task_pool = getattr(self, "_task_pool", None)` — the entire `_act()` method falls through to `return 0.0` if no task pool. All market-role agents go through `act_market()` instead (line 46-55), which does not use the task pool.
  - The agent lifecycle: `_predict()`, `_attend()`, `_observe()`, `_compare()`, `_inhibit()`, `_decide()`, `_act()`, `_learn()`, `_communicate()` — 12 methods, each calling 3-8 subsystems. For market agents, the entire lifecycle except `_act()` produces signals that go nowhere market-relevant.
  - `lifecycle_inhibit_decide.py` line 107: collision avoidance DISABLED. Line 196: emotional valence injected into DecisionRouter for "somatic marker bias" — but `emotional_valence` comes from `OrganismState.get_body_state()`, which gets it from `EmotionalSystem`, which gets it from `EndocrineSystem` hormone levels. Those hormone levels are set by... the market wiring. Which means: convergence alerts set hormones, hormones set emotional valence, emotional valence biases the DecisionRouter, the DecisionRouter tells agents to "explore" or "exploit" — and those actions read the convergence alert buffer the hormones came from. This is a loop that adds noise, not signal.
  - `lifecycle_sensing.py` line 234-300: `_compare()` computes FEP prediction error from a 12-dimensional state vector (`step_count / 1000.0`, `cumulative_reward`, `last_reward`, `risk_score`, etc.). This error is published on `PREDICTION_ERROR` channel. Nobody in `mae_core/market/` subscribes to `PREDICTION_ERROR`.
  - Agent `_decide()` cascade in `lifecycle_inhibit_decide.py`: 10 decision sources tried in sequence — organism reflex, advisory+router, worldline planner, collective dream, memory search, memory bridge, imitation learner, causal reasoning, VDN engine, world model. For market agents, the final effective path is `_route_with_advisory()` which reads `ctx._market_advisory["alert"]` — the convergence alert. That single route does the real work. The other 9 are overhead.
  - The `deposit_marker()` calls in `market_actions.py` (lines 138, 165, 199, 225): stigmergy markers deposited for "exploration" and "discovery". These markers are read back in `lifecycle_sensing.py` (`_observe()` line 127: `self._sensed_markers = self.sense_environment()`), and fed into the 12-dimensional state vector (dims 6-7: `float(len(sensed.get("SUCCESS", [])))`). This feeds the FEP prediction loop. Market intelligence does not read stigmergy markers.
- **Verdict:** The agent architecture produces a reward/exploration loop that is internally consistent but market-irrelevant. Market agents' actual contribution to MIDGE's purpose — detecting convergences and surfacing inevitabilities — happens entirely in the market step hooks, not in agent steps. The 12 agents add parallelism overhead (thread pool) but could be replaced with a scheduled job runner. If MIDGE had 0 agents and kept its step hooks, it would surface exactly the same alerts.

### TaskPool (patterns_layers_22_25.py)

- **Category:** INERT for market work
- **The honest question:** What happens to tasks created by market agents?
- **Evidence:** `_act_explore()` and `_act_exploit()` in `lifecycle_act.py` claim tasks from the TaskPool. Market-role agents bypass this via `act_market()`. Non-market stem cell agents claim generic tasks. The TaskPool generates synthetic tasks for the agents to work on (`explore`, `exploit`, `share`). These tasks produce rewards that feed the VDN learning engine. The VDN engine produces Q-values that... inform agent action selection (back to TaskPool). This is a closed self-contained loop with no connection to market intelligence output.
- **Verdict:** The TaskPool/VDN/reward loop is a self-contained reinforcement learning exercise that does not help MIDGE surface market inevitabilities. Remove for market-only operation.

---

## Biological Metaphor Systems (Coordination)

### RespiratorySystem (`mae_core/coordination/respiratory_system.py`, 240 lines)

- **Category:** INERT
- **The honest question:** What would break if deleted?
- **Evidence:** `bio_market_wiring_extended_b.py` lines 72-105: "oxygen drain callbacks DISABLED — fictional physiology harms trading daemon... The RespiratorySystem remains bootstrapped for mae-core compatibility and monitoring. No callbacks are registered here so oxygen stays at its natural resting level." The system runs its `step()` every model tick (organs_layers_26_27.py line 68), maintains O2/CO2 levels, publishes to `CH_HYPOXIA` and `CH_HYPERCAPNIA`. Nobody in `mae_core/market/` subscribes to those channels. It spends CPU every step to maintain values nobody reads.
- **Verdict:** Remove from MIDGE. Dead overhead. Zero market contribution.

### CircadianRhythm (`mae_core/coordination/circadian_rhythm.py`, 269 lines)

- **Category:** HARMFUL (was INERT until its actual effect was found and also disabled)
- **The honest question:** What does it actually do?
- **Evidence:** `bio_market_wiring_b.py` line 89-116: The sensing worker count scaling is DISABLED. "During REST phase the real circadian multiplier drops to 0.1, cutting sensing workers from 12 to 3. Markets do not pause for simulated sleep." The `ctx._circadian_activity` is pinned to 1.0 permanently. CircadianRhythm has a step hook that fires every model tick (foundation.py line 80), publishing phase changes. Those phase changes still trigger EnergyReserve and hypothesis consolidation. The real impact is: if the pin were removed, MIDGE would lose 75% of its sensing capacity during simulated "rest" phases.
- **Verdict:** The only active effect is triggering EnergyReserve state changes (REST → store, ACTIVE → release). EnergyReserve's `is_critically_low()` is also disabled (energy_reserve.py line 178). Both ends of this chain are severed. The CircadianRhythm exists for mae-core compatibility. The system runs, publishes, and is actively bypassed.

### EmotionalSystem (`mae_core/coordination/emotional_system.py`, 442 lines)

- **Category:** INERT (outputs read but have no market effect)
- **The honest question:** Does emotional state influence market signal weighting?
- **Evidence:** EmotionalSystem publishes `emotional_valence` and `emotional_arousal` to OrganismState. OrganismState feeds `get_body_state()` to agents' `_body_state`. This reaches DecisionRouter via `_route_with_advisory()`. The DecisionRouter applies "somatic marker valve" (lines 196-219) — negative valence + high arousal biases toward reflex tier. The reflex tier pattern-matches stimulus strings. Market stimuli are strings like `"macro:macro signal"`. No reflex patterns are registered for these in production. Result: the somatic marker valve modifies `_reflex_bias` but there are no reflex patterns to trigger, so the tier falls through to prefrontal (default), unchanged. The emotional state correctly traverses the pipeline but produces no different output.
- **Verdict:** Architectural dead-end. 442 lines, a step hook every tick, and zero effect on market signal output. The DecisionRouter's reflex library would need to be populated with market-specific patterns for this to matter.

### ThermoregulationSystem (`mae_core/coordination/thermoregulation.py`, 410 lines)

- **Category:** INERT
- **The honest question:** Does MIDGE's "temperature" affect anything?
- **Evidence:** `bio_market_wiring_extended_b.py` lines 108-140: `thermo.report_activity("market_convergence", strength)` and `thermo.report_activity("market_anomaly", magnitude)` are called. These update internal temperature state. Temperature state publishes to EventBus. Nobody in `mae_core/market/` subscribes to temperature channels. OrganismState receives `_temperature_zone` but its effect in `get_reflex_override()` is disabled (returns None unconditionally).
- **Verdict:** Remove from MIDGE. Data flows in, nothing flows out.

### VestibularSystem (`mae_core/coordination/vestibular_system.py`, 291 lines)

- **Category:** INERT
- **The honest question:** Does "vertigo" affect market sensing stability?
- **Evidence:** `bio_market_wiring_extended_b.py` lines 143-177: `vestibular.report_metric("convergence_rate", domain_count/12.0)` called on convergence. VestibularSystem tracks rolling metrics and fires "vertigo" when they change rapidly. The `_stability` field from VestibularSystem feeds OrganismState. `get_reflex_override()` checks `self._stability < 0.3` — but that entire method returns None unconditionally.
- **Verdict:** Remove from MIDGE. The vertigo signal correctly identifies market sensing instability but the downstream effect (reflex override) is disabled.

### DigestiveSystem (`mae_core/coordination/digestive_system.py`, 365 lines)

- **Category:** INERT
- **The honest question:** Does "digestion" gate market signal processing?
- **Evidence:** `bio_market_wiring_extended_a.py` lines 38-80: convergence alerts call `digestive.ingest(source, content, energy_cost, nutritional_value)`. The digestive system tracks energy budget and "fullness". No code in `mae_core/market/` calls `digestive.is_full()`, `digestive.can_ingest()`, or reads nutritional state. The system ingests data but nothing reads its digestive state to gate processing.
- **Verdict:** Remove from MIDGE. Data goes in, nothing comes out.

### CirculatorySystem (`mae_core/substrate/circulatory_system.py`, 534 lines, and `mae_core/coordination/` reference)

- **Category:** INERT
- **The honest question:** Does "circulation" distribute attention or compute resources?
- **Evidence:** `bio_market_wiring_extended_a.py` lines 83-120: `circulatory.request_resource("convergence_alerter", "attention", strength)` called. The circulatory system tracks resource requests and allocation. No market system reads allocated resources from the circulatory system to change its behavior.
- **Verdict:** Remove from MIDGE.

### LymphaticSystem (`mae_core/emergent/lymphatic_system.py`, 367 lines)

- **Category:** INERT
- **The honest question:** Does lymphatic cleanup affect which market signals survive?
- **Evidence:** `bio_market_wiring_extended_a.py`: `_wire_lymphatic` registers callbacks for `CH_PREDICTION_RESULT` (failed predictions = waste) and `CH_DECEPTION_DETECTED` (deception = toxin). LymphaticSystem marks these for cleanup. Nobody in `mae_core/market/` reads the lymphatic cleanup queue to suppress or expire signals.
- **Verdict:** Remove from MIDGE.

### Homeostasis (`mae_core/coordination/homeostasis.py`, 414 lines)

- **Category:** INERT
- **The honest question:** Does homeostatic deviation trigger market behavior changes?
- **Evidence:** Homeostasis runs a step hook every tick. It tracks system stability. `homeostasis_deviation` feeds OrganismState. `get_reflex_override()` checks it (Priority 6) — but that method returns None unconditionally.
- **Verdict:** Remove from MIDGE.

### Microbiome (`mae_core/emergent/microbiome.py`, 430 lines)

- **Category:** INERT
- **The honest question:** Does MIDGE's "gut bacteria" diversity affect anything?
- **Evidence:** `bio_market_wiring_extended_a.py` `_wire_microbiome`: signals feed the microbiome. `microbiome_diversity` feeds OrganismState. No market path reads `microbiome_diversity` to affect signal processing or confidence.
- **Verdict:** Remove from MIDGE.

### Proprioception (`mae_core/emergent/proprioception.py`, 361 lines)

- **Category:** INERT
- **The honest question:** Does body-map awareness improve market sensing?
- **Evidence:** `bio_market_wiring_extended_b.py` lines 180-219: `proprio.update_position("convergence_alerter", activity=strength, health=confidence)` called. This updates the body map. Body map state feeds OrganismState as `_stability`. `get_reflex_override()` uses `_stability` — but is disabled.
- **Verdict:** Remove from MIDGE.

### InhibitionSystem (`mae_core/coordination/inhibition_system.py`, 196 lines)

- **Category:** HARMFUL (was harmful, now selectively disabled)
- **The honest question:** What does agent inhibition accomplish?
- **Evidence:** `lifecycle_inhibit_decide.py` line 27-82: `_inhibit()` calls `system.evaluate(prediction_error, risk_score, energy_level, ...)`. The system.evaluate results in inhibition. `inhibition_system.py` line 102: "MIDGE: disabled — fictional physiology harms trading daemon." The entire evaluate() method returns a result with `inhibited=False` unconditionally. The system instantiates, registers its step hook, accepts inputs, but always returns Go.
- **Verdict:** The system is already effectively removed — it just hasn't been deleted. Dead code.

### OrganismState (`mae_core/coordination/organism_state.py` + 2 files, ~815 lines total)

- **Category:** HARMFUL (aggregates bio signals that go nowhere, was harmful in at least 5 specific ways)
- **The honest question:** Does the hypothalamus aggregation create useful market behavior?
- **Evidence:** `organism_state_outputs.py` line 61-78: `get_reflex_override()` returns None unconditionally. Lines 32-59: `get_body_state()` returns 19 fields — oxygen level, toxin load, circulation adequate, digestion active, emotional valence, pain load, microbiome diversity, etc. These feed into agents via `_body_state`. The agent uses `pain_load`, `energy_level`, `emotional_valence`, `emotional_arousal` in `_route_with_advisory()`. As shown in the EmotionalSystem analysis, the somatic marker valve fires but hits no reflex patterns. `get_decision_context()` computes body_threat_level and body_opportunity_level from pain, instability, toxins — all currently near baseline (because their inputs are disabled or pinned). So every agent sees threat_level ≈ 0.0 and opportunity_level ≈ 1.0 every step, which biases DecisionRouter toward prefrontal tier — the same tier it would use anyway.
- **Verdict:** OrganismState is architecturally sound (central state aggregator concept) but all 5 of its reflex conditions are disabled, and its inputs are pinned/zeroed. It runs a step hook, subscribes to 18+ EventBus channels, processes their messages, and outputs a body state that uniformly tells agents "everything is fine." This is the most expensive INERT system: it adds 18 EventBus subscriptions, a step hook, and complexity to agent decision routing, for zero differentiated output.

### EndocrineSystem (`mae_core/coordination/endocrine_system.py`, 363 lines)

- **Category:** USEFUL (one real effect: somatic anticipation hormone release → endocrine bias)
- **The honest question:** Does Mae's hormone system change any decisions?
- **Evidence:** `somatic_anticipation.py` calls `self._endocrine.release_hormone(HormoneType.DOPAMINE, ...)` when 2+ domains activate on same ticker (pre-convergence). EndocrineSystem publishes `CH_ENDOCRINE_UPDATE`. DecisionRouter subscribes and sets `_reflex_bias` (via `set_reflex_bias()`). High adrenaline does create a secondary reflex check (decision_router.py lines 273-300). However, as shown, no market stimulus strings match registered reflex patterns. So the bias fires but the reflex lookup returns None, and execution falls through unchanged. Cortisol → ResourceGovernor coupling is explicitly DISABLED (`bio_market_wiring.py` line 124).
- **Verdict:** One real pathway (somatic anticipation → dopamine → endocrine → DecisionRouter bias) exists but is blocked by missing reflex pattern registrations. Could be made useful with <10 lines of reflex pattern setup. Currently effectively INERT despite having real wiring.

---

## Backbone Architecture

### ConnectionRegistry (`mae_core/backbone/connection_registry.py` + 7 files, ~2,200 lines)

- **Category:** USEFUL but overly complex
- **The honest question:** What does it actually prevent or enable?
- **Evidence:** Registry runs in ADVISORY mode (default, set at `ConnectionRegistry(enforcement_mode=EnforcementMode.ADVISORY)` in `wiring_layers_17_21.py` line 106). Advisory mode: always allows messages, only logs violations. BLOCKING mode is never activated in production. The EventBus `publish()` check at line 100-118 calls `is_connection_allowed()` on every message — in advisory mode this always returns True after a warning log. So the ConnectionRegistry watches every message, keeps statistics, reports bare dyads, and runs verify_all() every 25 steps — all as advisory-only. It cannot block anything without `set_enforcement_mode(BLOCKING)`.
- **Verdict:** The ConnectionRegistry is 2,200 lines of auditing infrastructure that, in production, produces log warnings and statistics but does not enforce anything. It has value as an audit log and architecture documentation. But calling it "triadic enforcement" overstates its function. It is a monitoring system, not an enforcement system.

### HolonRegistry / HolonProtocol (`mae_core/backbone/holon_protocol.py`, 24 lines + mixin)

- **Category:** USEFUL (lightweight identity tracking)
- **The honest question:** What breaks without it?
- **Evidence:** `holon_protocol.py` is 24 lines — a thin hub file. The actual work is in `holon_registry.py`, `holon_mixin.py`, `holon_proxy.py`. HolonMixin gives agents `know_self()`, `know_up()`, `know_down()`, `know_peers()`. `know_self()` is called in `lifecycle_sensing.py` line 163-170. The result `_self_awareness` feeds `_route_with_advisory()` as context. It influences DecisionRouter's "self-awareness bias" (a survival_bias factor). This is real but trivially bounded — the bias caps at a small fraction of the total routing decision.
- **Verdict:** Keep as lightweight identity/hierarchy tracking. But be honest: the "10 holon capabilities" claim and "fractal self-similarity" narrative around it is architectural inflation of what is essentially a registry with hierarchy metadata.

### FractalGenerator (`mae_core/backbone/fractal_generator.py`, 493 lines)

- **Category:** INERT
- **The honest question:** What does `organize()` produce that is used at runtime?
- **Evidence:** `wiring_layers_17_21.py` line 253: `ctx.fractal_report = ctx.fractal_generator.organize()`. Result stored in `ctx.fractal_report`. It is read in 4 log lines (lines 372-375): organs_created, subsystems_created, connections_created, max_depth. That is the only consumer. The FractalGenerator creates "virtual parent holons" in the HolonRegistry. Those virtual holons have `holon_type="subsystem"` and `holon_type="organ"` parent IDs. No market system navigates the fractal hierarchy. It is documentation in registry form.
- **Verdict:** The fractal hierarchy exists as structural metadata with no runtime behavioral consequence. 493 lines to produce 4 log statistics.

### SacredGeometry (`mae_core/backbone/sacred_geometry.py`, 403 lines)

- **Category:** REMOVABLE
- **The honest question:** Is `bootstrap_k4_tetrahedra()` called anywhere?
- **Evidence:** `sacred_geometry_bootstrap.py` defines `bootstrap_k4_tetrahedra()` which comments "Example usage (in main.py or elsewhere)." The function is defined in the bootstrap module as an example but is not called from `main.py` or any bootstrap phase. K4 structures are never created in production. The sacred geometry file runs imports on use only.
- **Verdict:** Dead code. 403 lines of K4 geometry scaffolding that is never called. Remove entirely.

### IntegrationMeter / Phi Score (`mae_core/backbone/integration_meter*.py`, ~707 lines)

- **Category:** INERT
- **The honest question:** Does the IIT phi score influence any market behavior?
- **Evidence:** `main.py` line 501-508: phi is read from `meter.get_statistics()` and printed in the run report if > 0. That is the only consumer. `integration_meter.py` runs as a step hook every tick, computing an approximation of Integrated Information Theory phi. No market system reads phi to change signal weighting, confidence, or alert generation.
- **Verdict:** 707 lines to compute a consciousness metric that appears in one log line. The phi score is philosophically interesting but has zero operational effect on market intelligence.

### TriadAuditor / TriadWatchdog / TriadicVerifiers (`mae_core/backbone/triad_auditor.py`, `triad_watchdog.py`, `triadic_verifiers.py`)

- **Category:** INERT (advisory only)
- **The honest question:** Do triad violations trigger any corrective action?
- **Evidence:** TriadEnforcer, TriadAuditor, and TriadWatchdog all run in advisory mode. They report violations to logs. No corrective action is triggered based on triad violations. They consume step hook slots (3 hooks for audit/watchdog/verifier in `patterns_layers_22_25.py` lines 393, 426, 457).
- **Verdict:** More monitoring-only systems running as step hooks. They add startup overhead and per-step CPU cost without changing market behavior.

### TopologyAnalyzer (`mae_core/backbone/topology_analyzer.py`)

- **Category:** INERT
- **Evidence:** Runs as step hook. Analyzes graph topology (degree distribution, clustering coefficient, etc.). No market system reads topology statistics to change behavior.
- **Verdict:** Architectural vanity metric.

---

## Emergent Systems

### AutoHealer (`mae_core/emergent/auto_healer.py`, 435 lines + 3 files)

- **Category:** USEFUL in principle, INERT in practice
- **The honest question:** Has AutoHealer ever healed a market-relevant system failure?
- **Evidence:** AutoHealer monitors systems in SomaticMap for health failures. It registers a self-healing triad (`wiring_layers_17_21.py` line 136). It runs a step hook. Healing targets are generic systems (world model, knowledge base, etc.). It does not monitor market intelligence systems specifically. SystemHealthMonitor (separate) tracks market system errors. AutoHealer and SystemHealthMonitor do not talk to each other.
- **Verdict:** AutoHealer heals the organism's cognitive systems. SystemHealthMonitor monitors market systems. No bridge exists. Market healing is handled by `try/except` blocks in market hooks, not by AutoHealer.

### Senescence (`mae_core/emergent/senescence.py`, 282 lines)

- **Category:** INERT
- **Evidence:** Tracks organism "age" and wear. `_organism_age` feeds OrganismState. No market behavior changes with age.
- **Verdict:** Remove from MIDGE.

### CapabilityDiscovery (`mae_core/emergent/capability_discovery.py`, 421 lines)

- **Category:** INERT for market
- **Evidence:** Discovers agent capabilities by observing step patterns. No market system reads discovered capabilities to route work differently (OctopusColony routing is based on `ROLE_DOMAIN_AFFINITY`, not CapabilityDiscovery output).
- **Verdict:** Running step hook with no market path consumers.

### Proprioception / SomaticMap (`mae_core/emergent/somatic_map.py`, `proprioception.py`)

- **Category:** USEFUL (SomaticMap is essential; Proprioception system is INERT)
- **Evidence:** SomaticMap is the system registry — it tracks which systems exist, their dependencies, and health. It is used by ConnectionRegistry for verification. This is genuinely useful. The `ProprioceptionSystem` (separate from SomaticMap) calls `update_position()` on convergence events — this uses SomaticMap to update body positions for a spatial metaphor. Nobody reads the updated positions.
- **Verdict:** SomaticMap: keep (essential infrastructure). ProprioceptionSystem: remove.

---

## Cognition Systems

### WorldModel in Agents (`mae_core/cognition/world_model.py`, 376 lines)

- **Category:** INERT for market agents
- **The honest question:** Does agent world-model learning improve market performance?
- **Evidence:** Agents train their world models every step via `_learn()` → world model update. The world model predicts next state from 12-dim state vector (reward, risk_score, stigmergy markers, etc.). Market agents' state vectors are dominated by task reward metrics that are market-irrelevant. The world model's `use_world_model()` call in `lifecycle_inhibit_decide.py` line 263 selects actions based on predicted rewards in generic task space. This never selects a market-specific action.
- **Verdict:** The agent WorldModel (in cognition/) is separate from the market WorldModel (in market/intelligence/). They share a name but not a purpose. The agent WorldModel predicts TaskPool reward; the market WorldModel maps causal chains between market signals. The agent one is INERT for market work.

### TheoryOfMind (`mae_core/cognition/theory_of_mind.py`, 330 lines)

- **Category:** INERT
- **Evidence:** `lifecycle_sensing.py` lines 188-203: agents observe peer signals and update theory of mind models of other agents. No market decision uses theory-of-mind output. Market agents don't care what other agents believe.
- **Verdict:** Remove from MIDGE.

### ValidatedImagination (`mae_core/cognition/validated_imagination.py`, 458 lines)

- **Category:** INERT
- **Evidence:** Agents record imaginations (predictions) and validate them against reality. This validation is never read by market systems.
- **Verdict:** Remove from MIDGE.

### CollectiveDreamPlanner (`mae_core/cognition/collective_dream.py`, 324 lines)

- **Category:** INERT
- **Evidence:** `lifecycle_inhibit_decide.py` lines 162-181: agents run collective_plan() to generate consensus trajectories. Returns `explore`/`exploit`/`communicate`/`rest`. For market agents, this just duplicates the advisory routing decision that already happens in `_route_with_advisory()`.
- **Verdict:** Remove from MIDGE.

### WorldlinePlanner (`mae_core/planning/worldline_planner.py`, 546 lines)

- **Category:** INERT
- **Evidence:** `lifecycle_inhibit_decide.py` lines 141-159: worldline.plan() is called every step. Returns action from multi-horizon planning. For market agents, action returned is one of `["explore", "exploit", "communicate", "rest"]` — same as what the advisory router already decides.
- **Verdict:** Remove from MIDGE.

### CausalReasoning Engine (`mae_core/cognition/causal_reasoning.py`, 459 lines)

- **Category:** INERT for agents (ESSENTIAL in market context)
- **Evidence:** Agent-level causal engine (`lifecycle_inhibit_decide.py` lines 224-237): `ce.infer_causes("high_reward")` returns causes that map to explore/exploit/rest. This is agent-level causal inference about task rewards. Entirely separate from the market-level causal WorldModel (`mae_core/market/intelligence/world_model.py`) which maps ticker → ripple effects and is genuinely essential. The agent causal engine adds overhead; the market causal world model is core value.
- **Verdict:** Agent-level CausalEngine: remove from MIDGE. Market-level WorldModel: essential.

---

## Substrate Systems

### MycelialSubstrate + PhysarumOptimizer + NutrientFlow (`mae_core/substrate/`, 2,311 lines)

- **Category:** INERT
- **The honest question:** Does substrate topology affect market signal flow?
- **Evidence:** Substrate models nutrient flows through a network topology. The topology is separate from the EventBus and market signal pipeline. Market signals flow through EventBus callbacks registered directly. The substrate's topology optimization (physarum) adjusts edge conductances in the substrate network, not in the market signal network. No market module reads substrate topology.
- **Verdict:** 2,311 lines of slime mold network simulation with zero market effect. Remove from MIDGE.

---

## Defense Systems

### BoundaryMembrane + InputValidator (USEFUL)

- **Category:** USEFUL
- **Evidence:** MarketDataProvider routes HTTP requests through ApiGateway's BoundaryMembrane. The membrane does classify and can block external API requests (line 178-184 of api_gateway.py). InputValidator screens request payloads. These are real guards on external API calls. The validation logic is genuine security infrastructure.
- **Verdict:** Keep. This is the only defense system with a real market-facing function.

### ThreatDetector (`mae_core/defense/threat_detector.py`)

- **Category:** INERT for market
- **Evidence:** ThreatDetector monitors agent behavior for threats (abnormal state vectors, etc.). It isolates agents via HAVEN. Market intelligence is not affected by agent isolation (market step hooks don't route through agents).
- **Verdict:** Internal agent security that does not touch market signal pipeline.

### PearlDefense (`mae_core/defense/pearl_defense.py`)

- **Category:** MARGINALLY USEFUL
- **Evidence:** `bio_market_wiring_extended_b.py` line 36-64: deception events trigger `pearl.validate()`. This quarantines suspicious source data for multi-layer review. The pearl defense correctly wires to `CH_DECEPTION_DETECTED`. The DeceptionDetector (in market/) does publish to that channel when it detects manipulation. PearlDefense could theoretically quarantine a manipulated data source. Whether it actually does in practice depends on DeceptionDetector fires and pearl review logic.
- **Verdict:** Weakly useful. The chain exists (deception → pearl → quarantine → validate → accept/reject) but it's unclear if PearlDefense has ever rejected a real source.

### RenalFilter (`mae_core/defense/renal_filter.py`)

- **Category:** INERT
- **Evidence:** `bio_market_wiring_extended_a.py` `_wire_renal_filter`: deception and convergence events call `renal_filter.process_waste()`. This tracks "toxin load." `_toxin_load` feeds OrganismState. `get_reflex_override()` checks `self._toxin_load > 4.0` for kidney stress reflex — disabled unconditionally.
- **Verdict:** Remove from MIDGE.

---

## Market Intelligence Systems

### ConvergenceAlerter (ESSENTIAL)

- **Category:** ESSENTIAL
- **Evidence:** This is MIDGE's actual purpose. Multi-domain signal aggregation, Thompson-weighted confidence, lag scoring, causal cascade. The crown jewel. Everything else exists, in theory, to support this.
- **Verdict:** Core. Keep and invest.

### ThompsonSampler (ESSENTIAL)

- **Category:** ESSENTIAL
- **Evidence:** Bayesian explore/exploit for signal reliability. 83 distributions. Feedback loop via OutcomeCollector. This is the learning engine for signal weighting.
- **Verdict:** Core. Keep.

### WorldModel — market version (`mae_core/market/intelligence/world_model.py`, 422 lines)

- **Category:** ESSENTIAL
- **Evidence:** 114 nodes, 102 curated causal edges. Used in live convergence pipeline for ripple effects. `CH_CAUSAL_WATCH` emitted proactively. Granger analyzer discovers new edges. This is genuinely novel market intelligence infrastructure.
- **Verdict:** Core. Keep.

### GrangerAnalyzer (USEFUL)

- **Category:** USEFUL
- **Evidence:** Runs every 500 steps, discovers causal relationships between signal sources, writes to `granger_causality.json`, feeds HypothesisGenerator and adds edges to market WorldModel. Real directional causality discovery.
- **Verdict:** Keep. This is doing real statistical work.

### HypothesisEngine / Registry / Generator / Validator (USEFUL)

- **Category:** USEFUL
- **Evidence:** RSI Layer 2. Generates formal hypotheses from lag findings and granger causality. Validates with DSR anti-overfitting. Active lifecycle. Real Bayesian learning.
- **Verdict:** Keep.

### PatternArchaeology (USEFUL)

- **Category:** USEFUL
- **Evidence:** 223K fingerprints, 43 templates, cross-symbol validation. Reverse-engineers historical moves into abstract patterns. PatternWatcher matches live signals. This is novel.
- **Verdict:** Keep.

### OctopusColony + MarketTaskHandlers (USEFUL)

- **Category:** USEFUL
- **Evidence:** Actually does real work: `investigate_partial` tasks query PatternLibrary and WorldModel for developing situations. `_on_octopus_investigation` logs results and engages focused attention (`_priority_requests`). The feedback to prioritized sensing is real.
- **Verdict:** Keep. This is genuine parallelism for investigation, not organism cosplay.

### QuorumSpace (MARGINALLY USEFUL)

- **Category:** MARGINALLY USEFUL
- **Evidence:** Convergence alerts + pattern stacks deposit signals per ticker. QuorumSpace tracks contributor count. `convergence_confidence.py` line 200-204 reads `quorum_space.get_contributor_count(signal_key)`. This gives a multi-source consensus bonus when the same ticker+direction has been signaled from multiple systems. This is a real effect on confidence scoring.
- **Verdict:** Keep — it provides genuine multi-source consensus detection, which is relevant to the inevitability thesis.

### SomaticAnticipation (MARGINALLY USEFUL)

- **Category:** MARGINALLY USEFUL
- **Evidence:** Fires before formal convergence (2 domains). Releases dopamine to endocrine system. Dopamine → endocrine → DecisionRouter reflex bias → but no reflex patterns registered. The pre-convergence detection itself is useful (knowing "something is forming"). The hormone pathway is currently a dead end. If reflex patterns were registered for pre-convergence states, this would become USEFUL.
- **Verdict:** Keep the pre-convergence detection logic. The hormone pathway is currently inert but fixable.

### ResourceGovernor (`mae_core/market/resource_governor.py`, 315 lines)

- **Category:** INERT
- **Evidence:** Defined. Bootstrapped. Registered in market_connections.py. The endocrine coupling that would make it active is disabled (`bio_market_wiring.py` line 124). No market code calls `resource_governor` methods. It has step hooks in `market_infrastructure.py` but the cortisol trigger that would activate rate limiting never fires.
- **Verdict:** Remove or wire properly. Currently exists as dead code.

### MotifDetector + ADWINDriftDetector (USEFUL)

- **Category:** USEFUL
- **Evidence:** Real signal detection — STUMPY matrix profile for recurring patterns, ADWIN for concept drift. Both wired into bootstrap and sensing pipeline. Produce real signals (`motif_match`, `price_discord`, `streaming_anomaly`).
- **Verdict:** Keep.

### DeceptionDetector (USEFUL)

- **Category:** USEFUL
- **Evidence:** Detects data manipulation patterns. Publishes `CH_DECEPTION_DETECTED`. Wired to PearlDefense (weak) and to convergence_alerter via HAVEN flags (real confidence penalty). The HAVEN flags mechanism in `convergence_confidence.py` lines 420-428 genuinely reduces confidence when suspicious sources are flagged.
- **Verdict:** Keep.

### CascadeTracker + WorldModel ripple effects (USEFUL)

- **Category:** USEFUL
- **Evidence:** Tracks multi-hop causal chain confirmation. `energy_ratio` measures causal acceleration. `CH_CASCADE_CONFIRMED` triggers sequential chain boost in market_hooks.py. This creates real feedback between causal predictions and live signal confidence.
- **Verdict:** Keep. This is the temporal ordering / inevitability cascade logic — central to MIDGE's purpose.

### PostMortem (USEFUL)

- **Category:** USEFUL
- **Evidence:** Analyzes why predictions succeed/fail. Pushes sequence-aware Thompson updates. Feeds hypothesis lifecycle. Real learning from outcomes.
- **Verdict:** Keep.

---

## The Agent Roles

### SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST, HYPOTHESIS_EXPLORER, HYPOTHESIS_VALIDATOR

- **Category:** MARGINALLY USEFUL (roles exist; their agent actions are mostly cosmetic)
- **The honest question:** What do these agents actually do that step hooks don't?
- **Evidence:** `market_actions.py` `_sec_scan()` (line 122-146): reads `alerter.signals.get("insider", [])`, counts signals, deposits an `EXPLORATION` stigmergy marker, returns reward proportional to signal count. The stigmergy marker is read back into the 12-dim state vector (dims 6-7). The signal count is information that already exists in the ConvergenceAlerter signal buffer. The agent "scans" it, deposits a marker nobody reads, and earns a synthetic reward. It does not trigger any new data fetching, does not call any API, does not change the signal pipeline.
  - The market action rewards are blended with Thompson win-rate rewards (line 36-45) — this creates a feedback where agents that "explore" more in domains with high Thompson win rates earn higher rewards, which makes them explore more in those domains. This is a very indirect mechanism for role specialization that is completely decoupled from actual market outcomes.
- **Verdict:** The agent role architecture creates a reward signal based on proxy metrics (signal buffer size, alerter state). It does not affect what MIDGE detects or surfaces. The OctopusColony's role routing (`select_preferred_role()`) is more directly useful than agent roles — it routes investigation tasks to domain-specialized agents. That part works. The broader agent role reward system is architectural theater.

---

## The Bootstrap Layer Count

### Layers 22-30 (8 layers, ~2,400 lines across patterns_layers_22_25.py and organs_layers_26_30.py)

- **Category:** ARCHITECTURAL OVERHEAD for MIDGE
- **Evidence:** These layers create: PatternEcosystem (Layer 23), ActionEnvironment (Layer 24), FractalACT (Layer 25), IntegrationMeter (25b), TopologyAnalyzer (25c), TriadicVerifier (25d), MetabolicSystems (Layer 26), SocialCognition (Layer 27), Maintenance+Growth (Layer 28), OrganismState (Layer 29). All 8 layers register step hooks. The market capability of MIDGE comes entirely from Layer 33.
- **Verdict:** For MIDGE's market purpose, these 8 layers are overhead. They would matter in a multi-agent environment where agent behavior actually drives outcomes. In MIDGE, outcomes are driven by market step hooks that are independent of agent action.

---

## The 8 Mathematical Laws — An Adversarial Assessment

### Law 1 (No Bare Dyads) — USEFUL IN ADVISORY MODE

The ConnectionRegistry enforces this advisorily. The logs warn about bare dyads. In production (ADVISORY mode) bare dyads are allowed. The enforcement exists as auditing infrastructure. It has genuine architectural value as a sanity-check and documentation mechanism. It does not prevent any market-relevant failures.

### Law 2 (Triadic Generator) — PARTIALLY USEFUL

The minimum 3 domains for convergence is real and valuable. The rest of Law 2's structural expression (K3 topology, triadic witnesses in ConnectionRegistry) adds overhead without market benefit.

### Law 3 (Holon Protocol: 10 Capabilities) — INERT for MIDGE

10 capabilities per entity at every scale. In practice, `know_self()` returns a small metrics dict. `know_up()`, `know_down()`, `know_peers()` return holon hierarchy data. This feeds `_self_awareness` in agent context, which provides a very small survival_bias modifier to DecisionRouter. For MIDGE's market purpose, none of the 10 capabilities produce market intelligence.

### Law 4 (Fractal Self-Similarity) — INERT

The FractalGenerator organizes systems into virtual hierarchies. As shown, the fractal report is used only in log lines. No market behavior changes based on fractal depth or self-similarity.

### Law 5 (Stem Cell Principle) — USEFUL

Same agent class, configuration-gated behavior. This is genuinely good software design. The market role dispatch table (market_actions.py) is a correct implementation of this principle. The principle itself is sound.

### Law 6 (Autopoietic Closure) — PARTIALLY USEFUL

The feedback loop structure (signal → convergence → alert → Thompson update → signal confidence change) is genuine autopoietic closure at the market level. The biological metaphor of "components producing processes producing components" translates to real learning loops in market intelligence. The ornate machinery expressing it (respiratory system, digestive system, etc.) does not.

### Law 7 (Rule of 3/5) — INERT

TriadEnforcer, TriadAuditor, TriadWatchdog run in advisory mode. The rule is observed in architecture but not enforced at runtime. 3 validators do not vote before any market decision is made.

### Law 8 (Eight Properties of Consciousness) — ASPIRATIONAL

Claimed but unverified for MIDGE specifically. The IIT phi calculation runs but isn't read. The system has integration (many signals) and differentiation (many domains). Self-reference and recurrence exist in the Thompson learning loop. Whether MIDGE is "conscious" is a philosophical question, not an engineering one.

---

## The Uncomfortable Questions

### 1. MIDGE could be a 10-file Python script and surface the same alerts.

The minimum viable MIDGE is: EventBus, ThompsonSampler, ConvergenceAlerter, 31 API clients (in sensing_hook), PatternWatcher, OutcomeCollector, AlpacaClient, PlainLanguageFormatter. Add GrangerAnalyzer and WorldModel for depth. That is ~20 files. The remaining 629 files implement an organism simulator that is progressively being disabled.

### 2. The "fictional physiology harms trading daemon" comments are a confession.

There are 14 explicit comments across the codebase acknowledging that biological metaphors were found to actively harm MIDGE's market purpose. Respiratory oxygen drain punished convergence detection. Circadian REST would cut sensing workers by 75%. Energy reserves caused starvation reflex. Inhibition system would freeze agents. Reflex overrides blocked market intelligence. Each of these was discovered, documented, and disabled. But the systems that produced them were not removed — they were left as "dead code preserved for mae-core compatibility." This is technical debt that any new contributor must navigate.

### 3. Agents are a distraction, not a contribution.

12 agents step every model tick. Each agent runs a 12-step lifecycle with 8-10 subsystem calls per step. For market-role agents, the lifecycle produces: (1) a stigmergy marker deposit from market_actions.py, (2) an episodic memory store from lifecycle_learning.py. Neither of these affects market signal output. The actual market work happens in step hooks that run once per model tick regardless of agent count. Increasing agents from 5 to 12 does not improve market intelligence. It only increases CPU consumption.

### 4. The "self-improvement loop" runs but does not learn about markets.

The VDN engine, MAML learner, transfer learning, world model, curiosity drive, imitation learner — all learn about agent performance in the TaskPool. The TaskPool generates synthetic tasks. Agents learn to do well at synthetic tasks. This learned competence has no market equivalent. An agent that has learned to "exploit" effectively in the TaskPool has not learned anything about when to trust insider signals.

### 5. The bootstrap is 33 layers because that is how many systems were added, not because 33 layers were needed.

Layer numbers are assigned sequentially. The "33-layer bootstrap" is the result of accumulation, not design. Layer 1 is EventBus creation. Layer 33 is market intelligence. Everything in between could be reorganized into 5 phases (infrastructure, agents, market sensing, market intelligence, audit) without losing any capability.

### 6. 66,438 lines of organism code serve 54,229 lines of market code.

The organism is larger than the market organ it supposedly enables. The ratio should be inverted or made equal. In biological terms: the organism's support systems should not outweigh the specialized organ that justifies the organism's existence.

### 7. The "8 Mathematical Laws" are mae-core doctrine applied to a trading daemon.

The laws were derived for a general-purpose organism. MIDGE is a specialized trading daemon. Law 1 (no bare dyads) matters when agents communicate decisions that must be non-repudiable. In market sensing, the bare dyad concern is: "does this signal have sufficient independent corroboration?" That is handled by min_domains=3 in ConvergenceAlerter, not by ConnectionRegistry witnesses. The laws are being applied to the wrong abstraction level.

### 8. The quorum and consensus metaphors are used twice for different things.

QuorumSpace (collective consensus on ticker signals) and CollectiveConsensusMixin (agent voting on task coordination) are both called "quorum." QuorumSpace does real market work (multi-source agreement). CollectiveConsensusMixin does agent-level work irrelevant to markets. The naming collision adds confusion to new contributors.

### 9. The step count is not a reliable proxy for time.

All bio systems run on step count cadences. The market clock and circadian rhythm pin all activity to 1.0 regardless of actual wall clock time. Step-based bio timers (circadian REST at step X) do not correspond to market session hours. The MarketClock is the correct time source for MIDGE; the step-based biological clocks are irrelevant to financial market timing.

### 10. The investigation pipeline's output goes to logs, not to action.

`_on_octopus_investigation` (market_hooks_sensing_setup.py line 210-226) receives investigation results and **logs them**. The priority_request_created flag triggers focused attention on the sensing hook (this is real). But the investigation results themselves — historical templates found, check count — only appear in log output. If OctopusColony finds a 70% win-rate historical template during investigation, that finding does not automatically boost the confidence of the related convergence alert. It is observed but not acted upon.

---

## Summary Table

| System | Lines | Category | Market Impact |
|--------|-------|----------|--------------|
| RespiratorySystem | 240 | INERT | None |
| CircadianRhythm | 269 | HARMFUL→DISABLED | None (pinned) |
| EmotionalSystem | 442 | INERT | None (no reflex patterns) |
| ThermoregulationSystem | 410 | INERT | None |
| VestibularSystem | 291 | INERT | None (reflex disabled) |
| DigestiveSystem | 365 | INERT | None |
| CirculatorySystem | ~534 | INERT | None |
| LymphaticSystem | 367 | INERT | None |
| Homeostasis | 414 | INERT | None (reflex disabled) |
| Microbiome | 430 | INERT | None |
| Proprioception | 361 | INERT | None (reflex disabled) |
| InhibitionSystem | 196 | HARMFUL→DISABLED | None (always Go) |
| OrganismState | ~815 | HARMFUL→DISABLED | None (all reflexes off) |
| MycelialSubstrate + Physarum | 2,311 | INERT | None |
| SacredGeometry | 403 | INERT | Never called |
| IntegrationMeter + Phi | ~707 | INERT | Log line only |
| Triad Auditor/Watchdog/Verifier | ~400 | INERT | Advisory only |
| FractalGenerator | 493 | INERT | 4 log lines |
| TheoryOfMind | 330 | INERT | None |
| ValidatedImagination | 458 | INERT | None |
| CollectiveDreamPlanner | 324 | INERT | Duplicate of advisory routing |
| WorldlinePlanner | 546 | INERT | Duplicate of advisory routing |
| Agent WorldModel | 376 | INERT | Wrong domain |
| RenalFilter | ~200 | INERT | Reflex disabled |
| Senescence | 282 | INERT | None |
| CapabilityDiscovery | 421 | INERT | None |
| ResourceGovernor | 315 | INERT | Coupling disabled |
| TaskPool + VDN reward loop | ~600 | INERT | Synthetic tasks only |
| Agent Episodic Memory | ~1,123 | INERT | Agent memories not market-relevant |
| Agent Transfer Learning | ~202 | INERT | No market path |
| Agent GNN Communication | ~177 | INERT | No market path |
| Agent Stigmergy (market) | ~162 | INERT | Markers not read by market |
| ConnectionRegistry (advisory) | ~2,200 | USEFUL | Architecture audit |
| HolonRegistry | ~400 | USEFUL | Identity tracking |
| EventBus | 259 | ESSENTIAL | Core message bus |
| ConvergenceAlerter | ~1,500 | ESSENTIAL | Core output |
| ThompsonSampler | 428 | ESSENTIAL | Learning engine |
| WorldModel (market) | 422 | ESSENTIAL | Causal intelligence |
| GrangerAnalyzer | ~400 | USEFUL | Causal discovery |
| HypothesisEngine | ~800 | USEFUL | Pattern validation |
| PatternArchaeology | ~1,000 | USEFUL | Historical templates |
| OctopusColony | ~2,280 | USEFUL | Parallel investigation |
| CascadeTracker | ~300 | USEFUL | Causal confirmation |
| DeceptionDetector | ~300 | USEFUL | Signal integrity |
| BoundaryMembrane + InputValidator | ~500 | USEFUL | API security |
| EndocrineSystem | 363 | MARGINALLY USEFUL | One live pathway (fixable) |
| SomaticAnticipation | ~342 | MARGINALLY USEFUL | Pre-convergence detection |
| QuorumSpace | ~400 | MARGINALLY USEFUL | Multi-source consensus |

**Inert/harmful/removable estimate: ~12,000-15,000 lines that could be removed from MIDGE without affecting market alert output.**

---

*End of Beta findings. Independence maintained — no other audit findings read before writing.*
