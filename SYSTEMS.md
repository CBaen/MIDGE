# MIDGE Systems

**Purpose:** Honest inventory of what exists, what emerges from connection, and what's theoretical.

---

## Existing Systems (85)

Built, tested, works independently. Grouped by module.

**Note:** SignalPriorityResolver is a per-agent subsystem (not registered on the backbone), but counted here as it's a complete, wired system with dedicated tests.

### Backbone (16 systems)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| EventBus | `backbone/event_bus.py` | In-process pub/sub messaging, 60+ channels | WIRED |
| StateStore | `backbone/state_store.py` | JSON-based key-value persistence | WIRED |
| VectorStore | `backbone/vector_store.py` | FAISS-backed semantic search | WIRED |
| TriadEnforcer | `backbone/triad_enforcer.py` | Rule of 3 majority voting enforcement | WIRED (called at bootstrap, advisory mode) |
| TriadWatchdog | `backbone/triad_watchdog.py` | Monitors validator bypass patterns | WIRED (periodic audit every 50 steps) |
| TriadAuditor | `backbone/triad_auditor.py` | Behavioral voting pattern analysis | WIRED (periodic audit every 50 steps) |
| TriadRegistry | `backbone/triad_registry.py` | Registers 16 processes with validators (CRITICAL=5, STANDARD=3) | WIRED (called at bootstrap) |
| HolonRegistry | `backbone/holon_protocol.py` | Fractal hierarchy tracker (containment: who's inside whom) | WIRED (populated at bootstrap) |
| HolonMixin | `backbone/holon_protocol.py` | Universal 10-capability self-awareness interface | WIRED (10th mixin on MycelialAgent) |
| ConnectionRegistry | `backbone/connection_registry.py` | Triadic witnessing for all 227+ connections (no bare dyads, 0 bare dyads), enforcement modes (PERMISSIVE/ADVISORY/BLOCKING) | WIRED (Layer 18, seal() at end of Layer 18, periodic verification) |
| AwarenessPulse | `backbone/holon_protocol.py` | Periodic hierarchy health check (orphans, health gradients, peer drift) | WIRED (step hook, interval=25, publishes holon.awareness_pulse + holon.anomaly_detected) |
| FractalGenerator | `backbone/fractal_generator.py` | Recursive fractal organizer (5 organs, 18 subsystems, K3 wiring with natural witnesses) | WIRED (Layer 20, publishes fractal.triad_created + fractal.organized) |
| IntegrationMeter | `backbone/integration_meter.py` | IIT Phi measurement + Markov blanket analysis at every scale | WIRED (Layer 25b, cadence 89, publishes integration.phi_measurement) |
| TopologyAnalyzer | `backbone/topology_analyzer.py` | Clustering coefficient, avg path length, small-world sigma, graph density | WIRED (Layer 25c, cadence 55, publishes topology.analysis) |
| WitnessNotifier | `backbone/witness_notifier.py` | Operational witnessing: witnesses receive shadow notifications when monitored connections fire | WIRED (Layer 15+, publishes witness.notification) |
| TriadicVerifier | `backbone/triadic_verifiers.py` | Verifies all 6 Part 1 mathematical proofs (Laman, Peirce, Hegel, Byzantine, Simmel, IIT) | WIRED (cadence 89, publishes triadic.verification) |

### Model & Agents (13 systems)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| MycelialModel | `model.py` | Mesa 3.4 orchestrator with backbone auto-creation | WIRED |
| BaseAgent | `agents/base_agent.py` | Thin Mesa agent with 5-phase lifecycle | WIRED |
| MycelialAgent | `agents/mycelial_agent.py` | Composed of 10 capability mixins (including HolonMixin) | WIRED |
| ConvergenceMixin | `agents/mixins/convergence.py` | Learning convergence detection | WIRED |
| GamificationMixin | `agents/mixins/gamification.py` | Motivation/achievement tracking | WIRED |
| SignalProcessingMixin | `agents/mixins/signal_processing.py` | Fast electrical signaling | WIRED |
| StigmergyMixin | `agents/mixins/stigmergy.py` | Pheromone trail markers | WIRED |
| GNNCommunicationMixin | `agents/mixins/gnn_communication.py` | Graph neural network routing with learning loop feedback | WIRED |
| TransferLearningMixin | `agents/mixins/transfer_learning.py` | Cross-task knowledge sharing | WIRED |
| EpisodicMemoryMixin | `agents/mixins/episodic_memory.py` | Experience replay integration | WIRED |
| CollectiveConsensusMixin | `agents/mixins/collective_consensus.py` | Quorum voting | WIRED |
| AdvancedFeaturesMixin | `agents/mixins/advanced_features.py` | World model + morphogenesis hooks | WIRED |
| StemCellRegistry | `agents/stem_cell.py` | Agent genome/epigenome tracking, 7 role profiles, redifferentiation | WIRED (Layer 21, publishes stem_cell.registered + stem_cell.redifferentiated) |
| MitosisMonitor | `agents/mitosis.py` | Autopoietic agent division: healthy agents produce new agents with mutated epigenomes | WIRED (publishes mitosis.division) |

### Memory (10 systems)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| MemoryCoordinator | `memory/coordinator.py` | Hub connecting all memory subsystems | WIRED (per-agent, publishes 5 channels, subscribers: CuriosityDrive, Endocrine, Morphogenesis) |
| EpisodicMemory | `memory/episodic_memory.py` | Prioritized replay buffer | WIRED |
| PrioritizedReplayBuffer | `memory/prioritized_replay_buffer.py` | SumTree O(log n) sampling | WIRED |
| SumTree | `memory/sum_tree.py` | Efficient priority tree | WIRED |
| SemanticRetriever | `memory/semantic_retriever.py` | FAISS semantic search | WIRED |
| ExperienceVAE | `memory/experience_vae.py` | VAE compression of experiences | WIRED |
| GenerativeReplayMemory | `memory/generative_replay.py` | Synthetic experience generation | WIRED |
| MemoryConsolidator | `memory/memory_consolidator.py` | Sleep-cycle offline learning | WIRED |
| WorkingMemory | `memory/working_memory.py` | 7+/-2 attention-gated buffer | WIRED |
| Experience | `memory/experience.py` | Data structure for transitions | WIRED |

### Learning (8 systems)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| FederatedRL | `learning/frl.py` | Peer policy sharing, trust-weighted | WIRED (P2P via EventBus) |
| VDN | `learning/vdn.py` | Value decomposition for credit assignment | WIRED (P2P via EventBus) |
| HAVEN | `learning/haven.py` | Byzantine fault detection, risk contagion | WIRED (publishes to EventBus) |
| CuriosityDrive | `learning/curiosity.py` | Novelty-based intrinsic motivation | WIRED (subscribes to memory.novel_experience, memory.experience_stored, memory.consolidation_complete) |
| TransferLearning | `learning/transfer_learning.py` | Cross-task knowledge bootstrapping | WIRED (injected with KnowledgeBase at bootstrap) |
| MAMLLearner | `learning/maml.py` | Meta-learning, few-shot adaptation | WIRED (injected with KnowledgeBase at bootstrap) |
| ImitationLearning | `learning/imitation.py` | Behavioral cloning, DAgger, GAIL | WIRED (shared, subscribes to frl.policy_update, SomaticMap registered) |
| KnowledgeBase | `learning/knowledge_base.py` | Shared skill repository | WIRED (EventBus, subscribes to frl.policy_update) |

### Cognition (5 systems)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| WorldModel | `cognition/world_model.py` | Imagination/prediction engine | WIRED (EventBus, publishes prediction events, per-agent + shared) |
| CollectiveDreamPlanner | `cognition/collective_dream.py` | Swarm imagination with weighted voting | WIRED (injected with shared WorldModel) |
| ValidatedImagination | `cognition/validated_imagination.py` | Prediction accuracy tracking | WIRED (instantiated at bootstrap) |
| DecisionRouter | `cognition/decision_router.py` | Three-tier brain: reflex/habit/deliberation | WIRED (EventBus, WorldModel, EndocrineSystem per-agent) |
| CausalReasoningEngine | `cognition/causal_reasoning.py` | Pearl's causal hierarchy | WIRED (EventBus subscriptions to temporal events, per-agent + shared) |

### Communication (15 systems)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| SignalBus | `communication/signal_bus.py` | Fast electrical signaling | WIRED |
| SignalPriorityResolver | `communication/signal_priority.py` | Thalamus-like signal triage (per-agent subsystem) | WIRED (BaseAgent step hook, MycelialAgent creates instance) |
| Stigmergy | `communication/stigmergy.py` | Pheromone trails with temporal decay | WIRED |
| QuorumSensor | `communication/quorum_sensor.py` | Quorum sensing main system | WIRED |
| QuorumSignal | `communication/quorum_signal.py` | Signal units for consensus | WIRED |
| QuorumSpace | `communication/quorum_space.py` | Aggregate signal space | WIRED |
| SpatialConsensus | `communication/spatial_consensus.py` | Location-aware voting | WIRED |
| ConsensusMetrics | `communication/consensus_metrics.py` | Voting statistics | WIRED |
| TemporalDecay | `communication/temporal_decay.py` | Time-based signal decay | WIRED |
| GNNCommunicator | `communication/gnn_communicator.py` | Graph neural network routing with active learning loop | WIRED (substrate topology sync, subscribes to topology events, RoutingOptimizer receives delivery outcome feedback) |
| GNNGraph | `communication/gnn_graph.py` | Network graph representation | WIRED |
| GNNMessage | `communication/gnn_message.py` | Message wrapper for GNN | WIRED |
| GNNPropagator | `communication/gnn_propagator.py` | Message propagation | WIRED |
| MessageAggregator | `communication/message_aggregator.py` | Vote-based deduplication | WIRED |
| PredictiveField | `communication/predictive_field.py` | Ambient coordination fields | WIRED (substrate position reading, step hook) |

### Coordination (3 systems)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| CircadianRhythm | `coordination/circadian_rhythm.py` | 3-phase clock (ACTIVE/CONSOLIDATION/REST) | WIRED (callback to EndocrineSystem in bootstrap) |
| EndocrineSystem | `coordination/endocrine_system.py` | 6-hormone modulation | WIRED (circadian phases, 8 hormone consumers: ThreatDetector, AutoHealer, CuriosityDrive, DecisionRouter, MemoryConsolidator, QuorumSensor, FRL, VDN) |
| OrganismState | `coordination/organism_state.py` | Integration hub: bridges 18 systems into agent behavior, activates dormant systems (WorldlinePlanner, CollectiveDreamPlanner, PredictiveField, MorphogenesisCoordinator), cross-system reflex coordination | WIRED (Layer 29, wired into MycelialAgent lifecycle: _decide(), _observe(), _learn()) |

### Defense (5 systems)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| ThreatDetector | `defense/threat_detector.py` | 4 biological defense strategies | WIRED (EventBus, HAVEN, SomaticMap, cortisol consumer, periodic scan) |
| InputValidator | `defense/input_validator.py` | Zero-trust boundary validation | WIRED (EventBus, trust propagation) |
| BoundaryMembrane | `defense/boundary_membrane.py` | Self/non-self recognition: classifies sources as self/trusted/provisional/quarantine. `register_self(names)` for internals, `register_source(name, trust)` for externals. Markov blanket permeability from IntegrationMeter. | WIRED (Layer 15+, pre-trusts all API providers at bootstrap via `register_source()`) |
| PearlDefense | `defense/pearl_defense.py` | Nacre-inspired threat encapsulation with alternating hard/soft validation layers | WIRED (wraps InputValidator, EventBus, step hook, SomaticMap registered) |
| TriageClassifier | `defense/triage_classifier.py` | Biological urgency classification: triages stimuli by biological need (threat, survival, social, learning). Assigns urgency and biologically-motivated recall weighting. | WIRED (EventBus, MemoryCoordinator, CuriosityDrive) |

### Emergent/Self-Improvement (3 systems)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| AutoHealer | `emergent/auto_healer.py` | Three-phase recovery (isolate/assess/restore) | WIRED (EventBus, substrate, causal engine, HAVEN, cortisol consumer, step hook) |
| CapabilityDiscovery | `emergent/capability_discovery.py` | Novel behavior detection | WIRED (EventBus, step hook, observation pipeline) |
| SomaticMap | `emergent/somatic_map.py` | Proprioception/body awareness | WIRED (EventBus, 36 systems registered at bootstrap) |

### Substrate (4 systems)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| MycelialSubstrate | `substrate/mycelial_substrate.py` | Network fabric with 4 topologies | WIRED |
| Topology | `substrate/topology.py` | Network topology generators | WIRED |
| NutrientFlow | `substrate/nutrient_flow.py` | Resource distribution | WIRED |
| PhysarumOptimizer | `substrate/physarum_optimizer.py` | Slime mold adaptive topology (Tero et al. conductance dynamics) | WIRED (reads NutrientFlow events, step hook, SomaticMap registered) |

### Morphogenesis (2 systems)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| MorphogenesisCoordinator | `morphogenesis/coordinator.py` | Growth engine, novelty spawning | WIRED (EventBus, OrganBuilder, model, substrate, SomaticMap, spawn_request + capability_found) |
| OrganBuilder | `morphogenesis/organ_builder.py` | Specialized team creation | WIRED (agent_factory creates real MycelialAgents) |

### Network/Octopus (5 systems)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| OctopusColony | `network/octopus_colony.py` | P2P peer network, Rule of 3 enforced | WIRED |
| OctopusAgent | `network/octopus_agent.py` | Individual octopus with health tracking | WIRED |
| OctopusCognition | `network/octopus_cognition.py` | Adaptive mode switching | WIRED |
| OctopusArm | `network/octopus_arm.py` | Autonomous limb with background threading | WIRED |
| OctopusSignals | `network/octopus_signals.py` | Signal enums and types | WIRED |

### Planning (2 systems)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| TemporalMemory | `planning/temporal_memory.py` | 4D spatiotemporal events | WIRED |
| WorldlinePlanner | `planning/worldline_planner.py` | Trajectory planning through spacetime | WIRED |

### External (1 system)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| ApiGateway | `external/api_gateway.py` | Mae's sensory gateway to the external world. Routes agent requests through BoundaryMembrane + InputValidator. Registered providers: Groq, Mistral, DeepSeek (LLM); MarketAux, Finnhub, AlphaVantage (financial data); Tavily (web search). Graceful: no keys = no providers = no-op. | WIRED (Layer 31, step hook, holon registered, 8 triadic connections) |

---

## Emergent Capabilities (10)

These capabilities would arise from connecting existing systems. The code exists on both sides -- they just need to talk to each other.

| # | Name | What Connects | What Emerges | Status |
|---|------|---------------|--------------|--------|
| 1 | Sleep-Wake Cycle | CircadianRhythm + EndocrineSystem + MemoryConsolidator | Phase changes trigger hormone shifts and memory consolidation during REST | WIRED (circadian->endocrine callback in main.py) |
| 2 | Memory-Driven Curiosity | MemoryCoordinator + CuriosityDrive | Novel experiences detected by memory feed curiosity's exploration bonus | WIRED (CuriosityDrive subscribes to memory.novel_experience) |
| 3 | Growth Response | MemoryCoordinator + MorphogenesisCoordinator | Memory capacity warnings trigger spawning of new specialist agents | WIRED (capacity_warning bridges to spawn_request in main.py) |
| 4 | Expertise-Weighted Imagination | CollectiveDreamPlanner + ValidatedImagination + WorldModel | Dream accuracy tracking weights expert opinions in collective planning | WIRED (CollectiveDream injected with WorldModel, ValidatedImagination instantiated) |
| 5 | Memory-Learning Pipeline | EpisodicMemory + FRL/VDN + MemoryConsolidator | Learning engines train on stored/consolidated experiences | WIRED (FRL has memory_coordinator for replay batches, consolidation triggers melatonin) |
| 6 | Topology-Aware Routing | MycelialSubstrate + GNNCommunicator | Messages route along actual network fabric topology, edge weights adapt from delivery outcomes | WIRED (GNNCommunicator receives substrate, subscribes to topology events, RoutingOptimizer learns from feedback) |
| 7 | Endocrine Behavior Modulation | EndocrineSystem + DecisionRouter + CuriosityDrive | Hormone levels modulate exploration, urgency, trust thresholds | WIRED (EndocrineSystem registers consumers: ThreatDetector, AutoHealer, CuriosityDrive, DecisionRouter, MemoryConsolidator, QuorumSensor, FRL, VDN) |
| 8 | Causal Imagination | CausalReasoningEngine + WorldModel | World model predictions validated against causal interventions | WIRED (CausalEngine subscribes to temporal events, feeds into WorldlinePlanner and AutoHealer) |
| 9 | Octopus Memory | OctopusAgent + EpisodicMemory + MemoryCoordinator | Octopuses remember task outcomes and improve over time | WIRED (OctopusAgent accepts memory_coordinator, stores outcomes, retrieves context) |
| 10 | Triad-Enforced Operations | TriadEnforcer + ConnectionRegistry + all CRITICAL/PROTECTED processes | Actions validated by majority vote before execution, connections enforced by registry (BLOCKING mode blocks bare dyads, unregistered, unhealthy) | WIRED (advisory mode on enforcement, blocking mode on connections, 16 processes registered at bootstrap) |

---

## Theoretical Systems (9)

Referenced in documentation or dependencies but zero implementation exists.

| # | Name | Where Referenced | What's Missing |
|---|------|-----------------|----------------|
| 1 | FastAPI External Interface | README.md, pyproject.toml (dependency) | No routes, no server code, no HTTP endpoints. Zero Python files. |
| 2 | Web Dashboard | README.md mentions observability | No HTML, no frontend code, no static files. |
| 3 | Domain Configuration | Mentioned conceptually | No centralized config module. Agents accept config dicts but no domain-level system. |
| 4 | Prometheus Metrics Export | pyproject.toml (dependency) | prometheus-client installed but no metrics defined or exported. |
| 5 | ~~GNN Learning Loop~~ | ~~GNN routing infrastructure exists~~ | DONE -- process_gnn_messages() reports delivery outcomes (success/failure/unhandled) to RoutingOptimizer. Edge weights learn via EMA. _communicate() override in MycelialAgent connects to agent step lifecycle. |
| 6 | Multi-Colony Coordination | OctopusColony is single-colony | Cross-colony communication not designed. |
| 7 | ~~Agent-Level Persistence~~ | ~~StateStore exists at model level~~ | DONE — serialize_state/restore_state on BaseAgent + all 10 mixins + MycelialAgent. |
| 8 | Inter-Octopus Arm Communication | OctopusArm has connected_arms attribute | No actual message passing between arms across octopuses. |
| 9 | ~~Signal Priority Protocol~~ | ~~Multiple communication channels exist~~ | DONE — SignalPriorityResolver triages concurrent signals with priority scoring, budget enforcement, tier mapping. |

---

## Tier 2 Persistence (Subsystem State)

Tier 1 persistence saves agent-level config and counters (serialize_state/restore_state on BaseAgent + mixins). Tier 2 persistence saves the actual learned knowledge inside subsystems.

### Pattern

Each subsystem implements two methods:
- `serialize()` -> `dict` -- returns all internal state needed to reconstruct the subsystem
- `restore(data: dict)` -- accepts a dict from serialize() and rebuilds internal state

### Covered Subsystems

| Subsystem | Module | Scope | What Is Persisted |
|-----------|--------|-------|-------------------|
| EpisodicMemory | memory | per-agent | Replay buffer experiences, priorities, SumTree |
| SemanticRetriever | memory | per-agent | FAISS index, stored documents |
| GenerativeReplayMemory | memory | per-agent | VAE weights, generated experiences |
| MemoryConsolidator | memory | per-agent | Consolidation stats, schedule state |
| MemoryCoordinator | memory | per-agent | Delegates to all child subsystems |
| WorldModel | cognition | per-agent | Ensemble weights, training history |
| ValueDecompositionEngine | learning | per-agent | Mixing network weights, value decomposition state |
| FederatedLearningEngine | learning | per-agent | Peer trust scores, policy cache |
| MAMLLearner | learning | per-agent | Meta-parameters, adaptation state |
| KnowledgeBase | learning | shared | Shared skill repository |

### Directory Layout

```
data/mae/subsystems/
  agents/{agent_id}/       # per-agent subsystem state
  shared/                  # shared systems (KnowledgeBase, etc.)
```

### Serialization Format

- Pickle: heavy objects (SumTree, numpy arrays, experience buffers)
- JSON: lightweight data (FRL trust scores)
- StateStore: metadata keys

### Wiring

- Restore: `main.py` after Layer 13 (agent creation), calls `load_subsystem_metadata()` then per-agent + shared restore
- Save: `model.shutdown()` iterates `_tier2_refs` dict, calls `save_subsystem_states()`
- Graceful degradation: missing files = start fresh (no crash on first boot)

---

## Nervous System Wiring (Agent Lifecycle)

Agents now have real sense-think-act-learn loops instead of stub lifecycle methods. This is the "nervous system" that connects perception, memory, decision-making, and communication into a coherent agent step.

### Agent Lifecycle Overrides (MycelialAgent)

| Method | What It Does |
|--------|-------------|
| `_build_state_vector()` | Assembles agent's current perception: position, energy, risk, reward history, hormone levels, nearby agent count |
| `_observe()` | Senses stigmergy markers within radius, searches episodic memory for similar past states |
| `_decide()` | Uses DecisionRouter (reflex/habit/prefrontal) with state vector + observations to choose action |
| `_learn()` | Stores experiences in episodic memory, deposits stigmergy markers (success/danger), periodic hippocampal replay every 10 steps |
| `_communicate()` | Processes incoming GNN messages (with delivery outcome feedback to RoutingOptimizer) |

### Signal Handlers Implemented (SignalProcessingMixin)

| Handler | Behavior |
|---------|----------|
| `_handle_opportunity_signal` | Reduces agent risk assessment based on signal strength |
| `_handle_collaboration_signal` | Checks own capabilities against requested ones, responds if match found |
| `_handle_knowledge_share_signal` | Blends incoming risk assessment with own using weighted average |

### Bugs Fixed

| File | Bug | Fix |
|------|-----|-----|
| `stigmergy.py` | `follow_trail()` passed invalid `attractive` kwarg to `get_gradient()` | Removed invalid parameter |
| `stigmergy.py` | `deposit_marker()` used `agent_id` (nonexistent) | Changed to `depositor_id` (correct field name) |
| `episodic_memory.py` | `store_experience()` called nonexistent `.store()` method | Uses `Experience` dataclass + `.add()` |
| `episodic_memory.py` | `not self.episodic_memory` was falsy for empty buffer, preventing all storage | Changed to `self.episodic_memory is None` |

### Tests

31 new tests in `tests/test_nervous_system.py` covering observation, decision, learning, signal handlers, stigmergy fixes, communication, and integration. Total: 1574 tests passing.

---

### Patterns (1 system + 1 subsystem)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| GlobalWorkspace | `patterns/global_workspace.py` | GWT competitive ignition: patterns compete via activation accumulation (EMA alpha=0.4), ignition threshold 0.7, suppression 0.3, 3-step refractory, triadic corroboration | WIRED (embedded in PatternCortex, EventBus broadcast on ignition) |
| AttentionalGate | `patterns/attentional_gate.py` | TRN-analog attentional gating: per-domain attention vectors, gating formula gated = raw * (0.5 + 0.5 * attention), surprise bypass >0.9, prediction error widens attention | WIRED (Layer 23, PatternBus integration, recurrent loop: sense -> cortex -> advisory -> gate -> sense) |

### Environment (1 system)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| TaskPool | `environment/task_pool.py` | Action environment: generates tasks for agents, step hook for task lifecycle | WIRED (Layer 24, injected into all agents, SomaticMap registered) |

### Fractal Action (1 system + 2 subsystems)

| Name | File | Description | Wiring Status |
|------|------|-------------|---------------|
| SubsystemAction | `backbone/fractal_generator.py` | Fractal ACT at subsystem level: delegates to child systems, aggregates results upward | WIRED (Layer 25, part of OrganismAction hierarchy) |
| OrganAction | `backbone/fractal_generator.py` | Fractal ACT at organ level: delegates to SubsystemActions, aggregates results | WIRED (Layer 25, part of OrganismAction hierarchy) |
| OrganismAction | `backbone/fractal_generator.py` | Fractal ACT at organism level: delegates to OrganActions, runs every 10 steps, top of action hierarchy | WIRED (Layer 25, step hook, registered in systems dict) |

---

**Last updated:** 2026-02-12
