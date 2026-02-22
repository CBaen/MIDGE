# Mae-Core Master Lifecycle Audit

**Date:** 2026-02-12
**Scope:** 16 audit teams, ~80+ sub-agents, all 12 lifecycle steps + 4 cross-cutting dimensions
**Duration:** Single session, parallel execution
**Prior Health Score:** 7.8/10 (from system audit same day)

---

## Executive Summary

**Overall Health: 6.4/10 (lifecycle depth) vs 7.8/10 (system wiring)**

The system wiring audit scored Mae 7.8/10 because connections work. This lifecycle audit goes deeper and finds the connections carry too little signal. Mae's architecture is impressively broad (73 systems, 7 communication channels, 9 learning subsystems) but shallow in execution. Many systems are built but not wired, wired but not consumed, or consumed but not at the right scale.

**Strongest:** ADVISE (A-) -- genuine GWT competitive ignition, 304 tests
**Weakest:** FRACTAL (C-) -- 9 of 10 holon capabilities exist only at agent level
**Biggest Gap:** No goal management, no inhibition, no precision-weighted attention (Team 15: grade F for goal management)

---

## Team Grades

| Team | Step/Dimension | Grade | Key Finding |
|------|---------------|-------|-------------|
| 0 | TRIAGE | C | Signal priority resolver works but no biological urgency classification |
| 1 | PREDICT | C+ | FEP prediction exists but PREDICTION_ERROR has zero subscribers |
| 2 | OBSERVE | B+ | Strong sensory integration, PatternSense z-score normalization solid |
| 3 | COMPARE | C- | Prediction error computed but orphaned -- nobody listens |
| 4 | DECIDE | B- | 3-tier cascade works but no Go/No-Go inhibition, somatic markers unused |
| 5 | ACT | B+ | Only lifecycle step with genuine fractal implementation |
| 6 | LEARN | B- | 5 of 9 learning subsystems are passive (never invoked), no trainable policy |
| 7 | COMMUNICATE | B- | 7 channels built, stigmergy write-only, quorum disabled, 10/17 files untested |
| 8 | ADVISE | A- | Full pipeline with GWT competitive ignition, strongest step |
| 9 | CONSOLIDATE | B+ | Dual consolidation (episodic + pattern), proper Fibonacci cadence |
| 10 | HEAL | B- | AutoHealer works but HAVEN protocol incomplete |
| 11 | RECALL | B- | 7 recall pathways but triadic verification not enforced on all |
| 12 | FRACTAL | C- | Law 4 most violated -- self-similarity only at agent level |
| 13 | CONSCIOUSNESS | B- | 6/8 properties partially present, none fully implemented |
| 14 | EMERGENT | B- | Quorum sensing disabled, all agents at (0,0) killing stigmergy |
| 15 | MISSING STEPS | C+ | 4 unanimous gaps identified across biology, CogArch, and active inference |

**Grade Distribution:** 0 A's, 1 A-, 3 B+, 5 B-, 1 C+, 2 C, 1 C-, 1 C+, 1 C- = weighted average ~B-/C+

---

## Critical Findings (Cross-Team Consensus)

### 1. PREDICTION_ERROR Is Orphaned (Teams 1, 3)
The COMPARE step computes prediction error and emits it to PREDICTION_ERROR channel. **Zero systems subscribe.** This is the core of Free Energy Principle -- prediction error should drive ALL learning and adaptation. Currently it drives nothing.

### 2. Law 4 (Fractal) Is Systematically Violated (Team 12)
9 of 10 holon capabilities (sense, remember, decide, learn, heal, know_self, know_up, know_down, know_peers) exist ONLY at the agent level. They do not repeat at subsystem, module, organ, or organism level. ACT is the sole exception with fractal implementation. This is Mae's most violated mathematical law.

### 3. 5 of 9 Learning Subsystems Are Passive (Team 6)
FRL (fractal reinforcement learning), VDN (value decomposition), MAML (meta-learning), Transfer learning, and Imitation learning are instantiated but their learn methods are never called in the lifecycle. Only WorldModel, TD-error, episodic replay, and spreading activation are active. No trainable policy network exists.

### 4. Stigmergy Is Write-Only (Team 7)
Agents deposit EXPLORATION, SUCCESS, and DANGER pheromone markers but `sense_markers()`, `sense_environment()`, and `follow_trail()` are never called in the lifecycle. The entire stigmergy feedback loop is broken -- markers are deposited into the void.

### 5. Quorum Sensing Disabled (Teams 7, 14)
`quorum_sensing_enabled` defaults to False. QuorumSensor objects are created per-agent and injected but the gating flag prevents any quorum behavior. Agents never differentiate into COMMUNICATOR or COORDINATOR roles.

### 6. All Agents at Position (0,0) (Team 14)
Spatial dynamics are dead. All agents exist at the origin, which means stigmergy gradients are meaningless, PredictiveField spatial coordination is collapsed, and spatial consensus has no variation.

### 7. Somatic Markers Wired But Not Consumed (Team 4)
EmotionalSystem produces somatic markers that are injected into decision routing context, but DecisionRouter never reads the `somatic_markers` field from context. The biological shortcut (gut feelings guiding decisions) is built but disconnected.

### 8. No Go/No-Go Inhibition (Teams 4, 15)
Mae MUST produce an action every tick. There is no mechanism to suppress responses. This violates Law 8 Property 7 (competition/selection must include "no action") and is unanimously identified as missing by biology, cognitive architectures, and active inference research.

### 9. No Goal Management (Team 15: Grade F)
Mae has zero goal persistence. No goal stack, no subgoaling, no impasse detection, no multi-step intentions. ACT-R has goal buffers, SOAR has universal subgoaling. This is the single largest architectural gap.

---

## Missing Lifecycle Steps (Team 15)

### 4 Unanimous Gaps (all 3 research streams agree)

| Missing Step | What It Does | Priority |
|-------------|-------------|----------|
| **INHIBIT** | Go/No-Go gate after COMPARE, before DECIDE. Can veto action entirely. | P0 (safety) |
| **GOAL** | Persistent goal stack with push/pop, impasse detection, subgoaling | P0 (biggest gap) |
| **ATTEND** | Precision-weighted attention. Modulates all downstream processing. | P0 (inference quality) |
| **BROADCAST** | GWT competitive broadcast to all systems. Implements consciousness. | P0 (Law 8 compliance) |

### Additional Proposed Steps

| Step | Priority | What It Does |
|------|----------|-------------|
| REGULATE | P0 | Yerkes-Dodson arousal, sympathetic/parasympathetic balance |
| GATE | P1 | Sensory gating, habituation, pre-attentive filtering |
| SIMULATE | P1 | Counterfactual reasoning, EFE computation, mental rehearsal |
| CHUNK | P1 | SOAR-style chunking, procedural memory compilation |
| INTEROPREDICT | P1 | Predict body state, interoceptive prediction error |
| DEFAULT | P2 | DMN mind-wandering, creative recombination |
| META | P2 | Metacognitive monitoring, confidence calibration |
| PROSPECT | P2 | Prospective memory -- "when X, do Y" |
| SLEEP STAGES | P2 | Split CONSOLIDATE into LIGHT/DEEP/REM phases |

---

## Mathematical Law Compliance

| Law | Name | Compliance | Worst Offenders |
|-----|------|-----------|----------------|
| 1 | No Bare Dyads | C+ | GNN messages, SignalBus, PredictiveField all dyadic |
| 2 | Triadic Generator | B | PatternSharer K3 compliant; most systems are not triadic |
| 3 | Holon Protocol | C- | Communication systems lack 10 capabilities, no HolonProxy on most |
| 4 | Fractal Self-Similarity | C- | WORST. Only ACT is fractal. 9/10 capabilities agent-only |
| 5 | Stem Cell Principle | B+ | Uniform agent class, role differentiation unused at runtime |
| 6 | Autopoietic Closure | C | GNN routing self-produces; stigmergy, signals, quorum do not |
| 7 | Rule of 3/5 | B- | PatternSharer and QuorumSensor enforce 3+; SignalBus does not |
| 8 | Eight Consciousness Properties | C+ | 6/8 partially present, none fully implemented, missing BROADCAST |

---

## Improvement Roadmap (Prioritized)

### Phase 1: Wire What Exists (fixes, no new architecture)

1. **Subscribe to PREDICTION_ERROR** -- Route to learning subsystems, decision modulation, healing triggers
2. **Enable quorum sensing** -- Set `quorum_sensing_enabled=True` in agent_config, add lifecycle calls
3. **Wire stigmergy sensing** -- Add `sense_markers()` to `_observe()`, use gradients in `_decide()`
4. **Activate passive learning subsystems** -- Wire FRL, VDN, MAML, Transfer, Imitation into `_learn()`
5. **Read somatic markers in DecisionRouter** -- Use emotional context for intuitive decision biases
6. **Assign spatial positions** -- Distribute agents on grid, enable meaningful spatial dynamics
7. **Register GNN message handlers** -- Wire handlers for COLLABORATION_REQUEST, STATE_UPDATE, VOTE

### Phase 2: Add Missing Steps (P0 -- new architecture)

8. **INHIBIT step** -- Go/No-Go gate between COMPARE and DECIDE
9. **GOAL step** -- Persistent goal stack with impasse detection
10. **ATTEND step** -- Precision-weighted attention after TRIAGE
11. **BROADCAST step** -- GWT competitive broadcast (cadenced every 3 steps)
12. **REGULATE step** -- Arousal regulation after LEARN

### Phase 3: Fractal Compliance (Law 4 remediation)

13. **Subsystem-level holon capabilities** -- sense/decide/learn at subsystem scale
14. **Organ-level communication** -- Organ-to-organ message passing
15. **Organism-level self-awareness** -- Mae-as-holon with full 10 capabilities

### Phase 4: Depth (P1 steps + advanced)

16. **Trainable policy network** -- Beyond WorldModel weight changes
17. **SIMULATE step** -- Counterfactual reasoning
18. **CHUNK step** -- Procedural memory compilation
19. **Test coverage** -- 10 of 17 communication files have zero tests

---

## Test Coverage Gaps

| Area | Dedicated Tests | Gap |
|------|----------------|-----|
| ADVISE pipeline | 304 | None (strongest) |
| Communication (3 of 17 files) | 85 | 10 files untested |
| Stigmergy | 0 | Complete gap |
| QuorumSensor/QuorumSpace | 0 | Complete gap |
| PredictiveField | 0 | Complete gap |
| NociceptionSystem | 0 | Complete gap |
| SignalBus | 0 | Complete gap |

---

## Comparison with Previous Audit

| Metric | System Audit (earlier today) | Lifecycle Audit (this session) |
|--------|------------------------------|-------------------------------|
| Focus | "Is it wired?" | "Is it correct, complete, fractal, biologically grounded?" |
| Health Score | 7.8/10 | 6.4/10 |
| Bugs Found | 12 (fixed 10) | 9 critical architectural gaps |
| Key Finding | 11 EventBus callbacks wrong arity | Law 4 systematically violated, goal management missing |
| Recommendation | Fix wiring | Add missing lifecycle steps, activate dormant subsystems |

---

## For the Next Instance

This audit identified that Mae's breadth (73 systems) far exceeds her depth. The architecture is impressively designed but underutilized. The most impactful work is NOT building new systems -- it's wiring the systems that already exist (Phase 1) and adding the 5 missing P0 lifecycle steps (Phase 2).

**Start with:** Subscribe to PREDICTION_ERROR (highest consensus across all teams), then INHIBIT step (highest priority from Team 15), then enable quorum sensing (simplest activation).
