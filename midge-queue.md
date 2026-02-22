# Mae-Core Queue

**Purpose:** Active tasks only. Git history preserves completed work.

---

## Completed (2026-02-11)

- [x] **Build PhysarumOptimizer** (2026-02-11)
- [x] **Build PearlDefense** (2026-02-11)
- [x] **Wire Octopus Memory** (2026-02-11)
- [x] **Wire ImitationLearning** (2026-02-11)
- [x] **Agent-level persistence** (2026-02-11)
- [x] **Integration test suite** (2026-02-11)
- [x] **Biological claims validation** (2026-02-11)
- [x] **Full organism wiring** (2026-02-11)
- [x] **Documentation audit and cleanup** (2026-02-11)
- [x] **Fractal architecture research** (2026-02-11)
      What: 5 parallel research agents on fractal networks, sacred geometry, consciousness math, autopoiesis, triadic principle
      Result: Synthesis at data/MAES-MATHEMATICAL-IDENTITY.md

---

## Completed (2026-02-18)

- [x] **Connect financial data + web search APIs** (2026-02-18)
      What: RestDataProvider (generic REST GET for MarketAux/Finnhub/AlphaVantage) + TavilyProvider (web search). Bootstrap Layer 31c registers all 4 providers gracefully. Fixed `test_bootstrap_no_api_key_graceful` (missing data key cleanup). 33 new tests.
      Result: 1970 tests pass. Mae can now sense financial markets and search the web via the existing oracle pathway.

---

## Active

### Fractal Architecture (Priority — from MAES-MATHEMATICAL-IDENTITY.md)

- [x] **1. Holon Protocol mixin** (2026-02-11)
      What: HolonRegistry + HolonMixin (10 capabilities). 10th mixin on MycelialAgent. 48 new unit tests, 11 integration tests.
      Result: 81 systems, 411 tests pass. Every agent knows self, parent, children, peers.

- [x] **2. Triadic connection enforcement** (2026-02-11)
      What: ConnectionRegistry with auto-witness assignment. 63 connections, 0 bare dyads. 28 unit tests, 6 integration tests.
      Result: 81 systems, 411 tests pass. Every connection has a triadic witness.

- [x] **3. Bidirectional awareness** (2026-02-11)
      What: HolonProxy + AwarenessPulse. 36 proxies injected (Layer 19). 25 unit + 3 integration tests.
      Result: 82 systems, 439 tests pass. Every system knows parent, children, peers. AwarenessPulse active.

- [x] **4. Fractal generator formalization** (2026-02-11)
      What: FractalGenerator backbone system. 4 organs, 13 subsystems, K3 wiring with natural witnesses. 25 unit + 7 integration tests.
      Result: 83 systems, 471 tests pass. 59 holons, 114 connections, 20-layer bootstrap.

- [x] **5. Stem cell agent refactor** (2026-02-12)
      What: AgentGenome (20 genes), AgentEpigenome, 7 ROLE_PROFILES, redifferentiate(), StemCellRegistry (Layer 21).
      Result: 84 systems, 507 tests pass, 60 holons, 114 connections, 21-layer bootstrap.

### Pre-Fractal Tasks

- [x] **Tier 2 persistence (memory contents)** (2026-02-12)
      What: serialize()/restore() on 8 per-agent subsystems (EpisodicMemory, SemanticRetriever, GenerativeReplayMemory, MemoryConsolidator, WorldModel, VDN, FRL, MAML) + 2 aggregation systems (MemoryCoordinator, KnowledgeBase). Wired into main.py (restore after Layer 13) and model.shutdown() (save via _tier2_refs). Fixed bug: main.py used _episodic (private) instead of .episodic (public), agents silently got None.
      Result: 534 tests pass (507 + 24 unit + 3 integration, 0 regressions). Persist dir: data/mae/subsystems/. Pickle for heavy objects, JSON for FRL trust, StateStore for metadata. Graceful degradation on missing files.

- [x] **Signal Priority Protocol** (2026-02-12)
      What: SignalPriorityResolver — per-agent thalamus. Queues signals between steps, sorts by composite score (priority×0.5 + urgency×0.3 + recency×0.2), coalesces same-type duplicates (log-boost), enforces per-step budget (10), maps to DecisionTier (Reflex/Habit/Prefrontal). Preemption for critical signals (≥0.9). Deferred overflow with age expiry. Wired into BaseAgent.step() and MycelialAgent.__init__. Also fixed latent bug: subscribe_to_signal passed agent_id kwarg that SignalBus.subscribe() silently rejected — standard handlers were never actually subscribed.
      Result: 562 tests pass (534 + 24 unit + 4 integration, 0 regressions). 84 systems, 60 holons, 114 connections, 21-layer bootstrap.

- [x] **Blocking Triad enforcement** (2026-02-12)
      What: EnforcementMode enum (PERMISSIVE/ADVISORY/BLOCKING) on ConnectionRegistry. Bootstrap grace period: starts PERMISSIVE, seal() transitions to configured mode. BLOCKING: rejects bare dyad registration (ConnectionError), disables unhealthy connections, is_connection_allowed() query API returns (False, reason). ADVISORY: logs + allows (current behavior preserved). TriadWatchdog escalates to ERROR in blocking mode. seal() called after Layer 18 register_all_connections(). Default remains ADVISORY for gradual rollout.
      Result: 591 tests pass (562 + 29 new: 27 unit + 2 integration, 0 regressions). Scope boundary: bus-level gating (SignalBus/EventBus checking connections) deferred as future task — requires system-to-agent ID mapping.

- [x] **GNN Learning Loop** (2026-02-12)
      What: Wired feedback from agent message processing to RoutingOptimizer. Modified process_gnn_messages() to report outcomes (success=priority reward, failure=-0.5, unhandled=0.1). Added _communicate() override in MycelialAgent. 30 new tests covering full send→route→receive→process→report→optimize cycle.
      Result: 621 tests pass (591 + 30 new, 0 regressions). Edge weights now learn from delivery outcomes via EMA.

- [x] **Wire the Nervous System** (2026-02-12)
      What: Implemented MycelialAgent lifecycle overrides (_build_state_vector, _observe, _decide, _learn, updated _communicate). Agents now sense stigmergy markers, search memory for similar states, deposit success/danger markers, do periodic hippocampal replay. Implemented 3 stub signal handlers in signal_processing.py (opportunity, collaboration, knowledge_share). Fixed bugs in stigmergy.py (invalid `attractive` param in follow_trail, `agent_id` → `depositor_id` in deposit_marker), episodic_memory.py (used Experience dataclass + .add() instead of nonexistent .store(), fixed truthiness bug: `not self.episodic_memory` → `self.episodic_memory is None`). 31 new tests.
      Result: 652 tests pass (621 + 31 new, 0 regressions). Agents have real sense-think-act-learn loops.

- [x] **Full codebase audit: fix pre-nervous-system bugs** (2026-02-12)
      What: 4-agent audit of every mixin↔subsystem interface found 9 bugs across 6 files. 4 crash bugs (wrong method signatures: sense_quorum, store_policy, store_value_function, use_world_model), 1 truthiness bug (empty buffer falsy in consolidate), 2 orphan subsystems (DecisionRouter + CausalEngine created but never injected), 1 stub producing garbage (_learn_from_batch random values), 1 dead code (store_episode called kb.clear() instead of kb.store_episode()). All fixed.
      Result: 675 tests pass (652 + 23 new, 0 regressions). Every mixin now correctly talks to its subsystem.

### Phase 1: Wire What Exists (from lifecycle audit)

- [x] **1. Subscribe to PREDICTION_ERROR** (2026-02-12)
      What: Added EventBus callback on signal.PREDICTION_ERROR in Layer 15. Routes high error (>0.7) to AutoHealer.report_anomaly(), boosts FRL learning rate for error >0.3.
      Result: 1574 tests pass. Core FEP signal no longer orphaned.

- [x] **2. Enable quorum sensing** (2026-02-12)
      What: Set quorum_sensing_enabled: True in agent_config. Added sense_quorum() call in _observe().
      Result: 1574 tests pass. Agents now sense collective density.

- [x] **3. Wire stigmergy sensing** (2026-02-12)
      What: Added follow_trail("SUCCESS"/"DANGER") in _observe(), danger gradient bias in _decide(). Strong danger (>0.5) returns "rest".
      Result: 1574 tests pass. Agents follow chemical gradients.

- [x] **4. Activate passive learning** (2026-02-12)
      What: Injected _frl_engine, _vdn_engine, _maml_learner, _transfer_engine, _imitation_learner into agents. Added 5 learning engine calls in _learn() at cadenced intervals (FRL/VDN every 10, MAML every 50, Transfer every 20, Imitation every 5).
      Result: 1574 tests pass. 5 dormant learning subsystems now active.

- [x] **5. Read somatic markers** (2026-02-12)
      What: Added somatic marker valve in DecisionRouter.route_decision(). Negative valence + high arousal biases toward reflex (fight/flight). Positive valence + high arousal dampens reflex (explore).
      Result: 1574 tests pass. Emotional context now informs decision tier.

- [x] **6. Assign spatial positions** (2026-02-12)
      What: Grid-based position assignment in Layer 12 agent loop. 100x100 grid with deterministic jitter. Sets agent.pos and stigmergy position.
      Result: 1574 tests pass. Agents no longer all at (0,0).

- [x] **7. Register GNN message handlers** (2026-02-12)
      What: 3 handler factories (COLLABORATION_REQUEST, STATE_UPDATE, VOTE) + registration in Layer 20. Collaboration checks capabilities, State feeds imitation learning, Vote feeds quorum sensor.
      Result: 1574 tests pass. 3 orphaned GNN message types now handled.

### Phase 2: Missing Lifecycle Steps

- [x] **8. INHIBIT step** (2026-02-12)
      What: InhibitionSystem (basal ganglia Go/No-Go). Evaluates prediction_error, risk, energy, somatic markers, goal priority, quorum pressure. Safety override after 5 consecutive inhibitions.
      Result: 1574 tests pass. New file: mae_core/coordination/inhibition_system.py

- [x] **9. GOAL step** (2026-02-12)
      What: GoalManager with GoalFrame stack, impasse detection (5 steps), subgoal creation, goal priority for decision modulation.
      Result: 1574 tests pass. New file: mae_core/cognition/goal_manager.py

- [x] **10. ATTEND step** (2026-02-12)
      What: _attend() wires AttentionalGate (existed in Layer 23) into agent lifecycle. Sets prediction error on gate, reads goal context for decision.
      Result: 1574 tests pass. AttentionalGate now injected into agents via Layer 30.

- [x] **11. BROADCAST step** (2026-02-12)
      What: _broadcast() every 3 steps. Agents submit salient signals (PE, reward, risk > 0.3) to EventBus for GWT competitive ignition.
      Result: 1574 tests pass. Agents now active GWT participants.

- [x] **12. REGULATE step** (2026-02-12)
      What: ArousalRegulator with Yerkes-Dodson target computation. _regulate() every 21 steps. Adjusts EndocrineSystem hormones toward optimal arousal.
      Result: 1574 tests pass. New file: mae_core/coordination/arousal_regulator.py

### Monolith Decomposition

- [x] **Decompose mycelial_agent.py** (2026-02-12)
      What: Extracted 16 lifecycle methods into 4 mixin files (SensingLifecycleMixin, DecisionActionLifecycleMixin, LearningLifecycleMixin, CommunicationLifecycleMixin). Core file: 339 lines.
      Result: 1574 tests pass. All files under 500 lines.

- [x] **Decompose main.py** (2026-02-12)
      What: Extracted 30-layer bootstrap into mae_core/bootstrap/ package (5 modules + context.py). Orchestrator: 245 lines. Uses SimpleNamespace context for cross-module state.
      Result: 1574 tests pass. All files under 602 lines.

### Complete the Circuits (Guiding Light directive)

- [x] **Law compliance audit** (2026-02-22)
      What: Triadic review of all 8 Laws. Composite score 8.0/10.
- [x] **Law 4 remediation** (2026-02-22)
      What: SubsystemAction/OrganAction/OrganismAction — 10 capabilities at every scale. MODULE_GROUPING + ORGAN_GROUPING. Law 4 ~8.5/10.
- [x] **HAVEN completion** (2026-02-22)
      What: 5 validator methods (validate_decision, validate_modification, validate_healing, validate_policy, validate_threat) + wire_triad_systems() for post-construction validator injection. 24 new tests.
- [x] **Triadic recall enforcement** (2026-02-22)
      What: HAVEN isolation filtering on MemoryBridge recall_peer_experiences (pre-filter + post-filter). EventBus publish on all 3 recall methods (memory.recall_completed). 3 triadic connections registered. 15 new tests.
- [x] **Communication test coverage** (2026-02-22)
      What: 5 new test files covering 7 previously untested communication modules (consensus_metrics, message_aggregator, quorum_signal_space, quorum_sensor, temporal_decay). 244 new tests.
      Result: 2304 tests pass. Communication module now substantially covered.
- [x] **TRIAGE biological urgency** (2026-02-22)
      What: TriageClassifier (4 medical categories: IMMEDIATE/URGENT/DELAYED/EXPECTANT). Reads nociception, threat_detector, endocrine. Injected into per-agent SignalPriorityResolvers. 3 triadic connections. 27 new tests.
      Result: 83 systems, 2405 tests pass. All "Complete the Circuits" tasks done.

### Sacred Geometry & Autopoietic Closure (2026-02-22)

- [x] **K4 Complete Graph** (2026-02-22)
      What: sacred_geometry.py — K4 tetrahedra built from K3 atoms. Edge, K3Triangle, K4CompleteGraph, K4Generator classes. All edges auto-witnessed (Law 1 by design). Bootstrap integration via sacred_geometry_bootstrap.py. 33 new tests.
      Result: Part 4 score ~3/10 → ~7.5/10. 2304 tests pass.

- [x] **Autopoietic Closure at higher scales** (2026-02-22)
      What: autopoietic_closure.py — AutopoieticMonitor base + SubsystemClosure/OrganClosure/OrganismClosure + ClosureCoordinator. Advisory-only. Fibonacci cadences (5/8/13). Monitors child health via HolonRegistry, recommends healing/restructuring. Bootstrap Layer 29a4 in organs.py. 29 new tests.
      Result: Law 6 now operational at all scales (was agent-only). 2304 tests pass.

### Infrastructure (lineage)

- [ ] **Step 7: Source routing + Tier 2 changes** (added: 2026-02-10)
- [ ] **Step 8: Update lineage-consult skill docs** (added: 2026-02-10)
- [ ] **Step 9: Build backtest suite** (added: 2026-02-10)
- [ ] **Step 10: Validate research pipeline** (added: 2026-02-10)

---

**When completing a task:**
1. Mark as `[x]` with completion date
2. Git commit preserves history -- no separate history file needed
