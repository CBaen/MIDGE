> Generated from 10-agent audit conducted 2026-02-11. ~50 sub-agents. Sources: biology papers, GitHub, research papers, full codebase trace.

# Mae Signal Path Audit: Grand Synthesis

## Executive Summary

Mae is a 23-layer, 90+ system organism implementing a fractal holarchy built from a triadic generator. Ten primary audit agents -- each commanding approximately five sub-agents -- traced every signal path through Mae's autopoietic loop: SENSE, ADVISE, DECIDE, ACT, LEARN, CONSOLIDATE, RECALL, HEAL, SELF-AWARENESS, and HIDDEN STEPS. This document synthesizes their findings.

**Overall Health: 5.2/10**

Mae's architecture is visionary and structurally sound. The 23-layer bootstrap in main.py wires together an impressive array of biologically-inspired subsystems: pattern sensing with triadic sharing, a thalamic relay (PatternBus), association cortex integration (PatternCortex), three-tier decision routing, episodic memory with prioritized replay, Qdrant-backed ancestral memory, immune-system-style healing, endocrine modulation, fractal hierarchy with holonic awareness, and more.

However, **the organism is not yet alive.** The signal path is complete from sense to decision, but:

1. **ACT is a stub.** `_act()` stores an action label and returns 0.0. No action changes the environment. The causal chain breaks at the most critical junction.
2. **Learning does not learn.** `_learn_from_batch()` computes TD errors but updates no weights. `consolidate()` calls methods that do not exist on the agent.
3. **Mae is reactive, not predictive.** Despite claiming FEP compliance, no prediction step occurs before observation. The WorldModel exists but is never called proactively.
4. **Many systems are wired but not connected.** CuriosityDrive, ValidatedImagination, FederatedLearning, ImitationLearning, PatternDistiller, CollectiveDream -- all are bootstrapped but their outputs are never consumed by the agent lifecycle.
5. **Fractal self-similarity is violated.** Each scale uses different architectures rather than recursive application of the same generator.

---

## Overall Health Scores by Step

| Step | Score | Key Finding |
|------|-------|-------------|
| SENSE | 6/10 | Functional signal pipeline. Lacks habituation, predictive coding, lateral inhibition, fractal self-similarity. |
| ADVISE | 6/10 | Good temporal integration. Lacks competitive ignition (GWT), prediction error, inhibitory gating. |
| DECIDE | 5/10 | Three-tier cascade works. Endocrine wiring is broken. Habit matching is exact-string-only. No GWT competition. |
| ACT | 1/10 | Two-line stub. Returns constant 0.0. No environmental effect. The weakest link in the entire system. |
| LEARN | 4.7/10 | Rich scaffolding. Core learning pathway incomplete: no weight updates, consolidation interface broken, disconnected subsystems. |
| CONSOLIDATE | 6/10 | Clean three-source extraction to Qdrant. No reconsolidation, no competitive selection, PatternDistiller is dead code. |
| RECALL | 5/10 | Seven recall pathways exist. Semantic recall has a live bug. Most pathways gated off by default. No triadic verification. |
| HEAL | 5/10 | Good 3-phase AutoHealer + HAVEN + SomaticMap. Cannot heal itself. Not fractal. AwarenessPulse anomalies go unheard. |
| SELF-AWARENESS | 4/10 | Know Self/Up/Down/Peers exist as interfaces. Never consulted during agent lifecycle. Static snapshots, not self-models. |
| HIDDEN STEPS | N/A | Identifies 14 missing capabilities. 6 are critical. The PREDICT step is the single most important addition. |

---

## The 5 Cross-Cutting Themes

These systemic patterns emerged across multiple audit reports independently:

### Theme 1: Reactive, Not Predictive (FEP Compliance Failure)

**Appeared in:** SENSE, ADVISE, LEARN, HIDDEN STEPS

Mae's mathematical identity explicitly claims Free Energy Principle compliance. The FEP requires organisms to predict before sensing, then compute prediction error as the primary learning and attention signal. Mae does not do this. The WorldModel exists and can predict, but it is only consulted as a fallback in `_decide()`, never proactively before `_observe()`. PatternSense detects but never predicts. PatternBus receives but never expects. The cortex generates advisories but never sends predictions downward.

**Impact:** This is the single largest gap between Mae's mathematical identity and her implementation. Every other gap is downstream of this one.

### Theme 2: Wired But Not Connected (Dead Code Paths)

**Appeared in:** DECIDE, LEARN, CONSOLIDATE, RECALL, HEAL, HIDDEN STEPS

Multiple systems are bootstrapped in main.py's 23 layers but their outputs are never consumed:
- CuriosityDrive computes intrinsic reward -- never added to agent reward
- ValidatedImagination publishes `cognition.imagination_validated` -- no subscribers
- PatternDistiller is injected into PatternConsolidator -- never called
- CollectiveDream publishes `cognition.collective_dream_complete` -- no subscribers
- EndocrineSystem calls `dr.set_reflex_bias()` -- method does not exist on DecisionRouter
- AwarenessPulse publishes `holon.anomaly_detected` -- no subscribers
- Peer experience recall exists in MemoryBridge -- never called from any decision path
- Multiple healing EventBus channels (`healing.started`, `healing.complete`, `healing.failed`) -- no subscribers

**Impact:** Significant capability is already built but invisible to the organism. Wiring these connections is high-value, low-effort work.

### Theme 3: Not Fractal (Scale Violations)

**Appeared in:** SENSE, DECIDE, CONSOLIDATE, HEAL, SELF-AWARENESS

The mathematical identity demands the same generator at every scale: cell, tissue, organ, organism. Every audit found that each scale uses a different architecture instead of recursive application of one pattern:
- SENSE: PatternSense (cell) uses 3 detectors; PatternSharer (tissue) uses triadic consensus; PatternBus (organism) uses domain grouping. Three different approaches.
- DECIDE: DecisionRouter exists only at agent level. No 3-tier routing at tissue, organ, or organism levels.
- HEAL: AutoHealer at organism level, holon_heal at agent level, but no tissue-level or organ-level healing.
- CONSOLIDATE: Agent consolidation uses replay; system consolidation uses pattern extraction. Different mechanisms.
- SELF-AWARENESS: holon_know_* exists at every level but is never called during the step lifecycle, making it structurally present but functionally absent.

**Impact:** The fractal self-similarity principle is the most consistently violated requirement of the mathematical identity.

### Theme 4: No Competition (GWT Compliance Failure)

**Appeared in:** SENSE, ADVISE, DECIDE, HIDDEN STEPS

Global Workspace Theory requires signals to compete for workspace access. Only "winning" coalitions should be broadcast. Most processing should be unconscious. In Mae:
- Every PatternSignal that enters the PatternBus inbox reaches the cortex.
- Every digest produces a PatternAdvisory.
- The DecisionRouter uses serial cascade (first match wins), not parallel competition.
- There is no ignition threshold, no competitive elimination, no unconscious processing layer.

**Impact:** GWT is listed as a core requirement in the mathematical identity. Its absence means Mae has no mechanism for selective attention or information prioritization.

### Theme 5: Self-Awareness Exists But Does Not Participate

**Appeared in:** SELF-AWARENESS, HEAL, HIDDEN STEPS

Mae has an impressive self-awareness infrastructure: HolonRegistry with 67+ registered holons, ConnectionRegistry with 114+ triadic connections, SomaticMap with blast radius analysis, AwarenessPulse with periodic health scans, and holon_know_self/up/down/peers on every holon.

None of it participates in the agent's cognitive loop. No agent consults its self-awareness during `_observe()` or `_decide()`. The self-model is a static snapshot, not a strange loop. The awareness data exists as queryable metadata but does not influence behavior. This is introspection without the loop -- the outgoing half with no return path.

**Impact:** Self-reference that influences own dynamics (Strange Loops) is a core requirement. Currently, self-awareness is decorative rather than functional.

---

## The Revised 13-Step Autopoietic Loop

Current loop: `Detect -> Advise -> Decide -> Act -> Learn -> Consolidate -> Recall`

Proposed complete loop with hidden steps identified by the audit:

```
1. PREDICT     - WorldModel generates expected next state/reward BEFORE sensing
2. ATTEND      - Top-down attention from goals + advisory + hormones gates signal processing
3. SENSE       - Observe environment: stigmergy, pattern sense, signal triage
4. COMPARE     - Compute prediction error (predicted - actual). The FEP surprise signal.
5. ADVISE      - PatternBus -> PatternCortex -> PatternAdvisory. Triggers hormonal response.
6. SELECT      - Goal-directed selection from advisory-guided DecisionRouter with GWT competition
7. INHIBIT/ACT - Go/no-go gate. Execute action OR suppress. Triadic: plan/execute/verify.
8. LEARN       - Store experience. Prediction error as reward component. Update weights.
9. REGULATE    - Endocrine response to step outcome. Active parasympathetic recovery.
10. COMMUNICATE - GNN messages, stigmergy, pattern sharing, learning broadcast to peers.
11. CONSOLIDATE - Periodic: memory consolidation + pattern distillation + ancestral storage.
12. RECALL      - Ancestral pattern recall informs next cycle's predictions.
13. DEVELOP     - Slow timescale: apoptosis, re-differentiation, fractal reorganization.
```

New steps relative to current: PREDICT (1), ATTEND (2), COMPARE (4), INHIBIT (7), REGULATE (9), DEVELOP (13).

---

## Priority Roadmap

### Tier 1: Critical (Fixes broken fundamentals)

| # | What | Why | Effort |
|---|------|-----|--------|
| 1.1 | **Implement action environment + override `_act()`** | ACT is a stub returning 0.0. Nothing in Mae changes the world. Without cause-effect power, learning, prediction, and everything downstream is meaningless. | HIGH |
| 1.2 | **Add PREDICT step before `_observe()`** | Transforms Mae from reactive to predictive. Satisfies FEP. WorldModel.predict() already exists -- it just needs to be called at the right time. | MEDIUM |
| 1.3 | **Fix `_learn_from_batch()` to actually update weights** | Learning is a no-op. The replay infrastructure is excellent but the endpoint is a placeholder. | MEDIUM |
| 1.4 | **Fix MemoryConsolidator interface mismatch** | `consolidate()` calls `get_learning_rate()`/`set_learning_rate()` which do not exist on BaseAgent. Dead code path. | LOW |
| 1.5 | **Fix endocrine-router wiring** | `set_reflex_bias()` does not exist on DecisionRouter. Adrenaline re-check is identical to first check. Two bugs making endocrine influence on decisions completely non-functional. | LOW |
| 1.6 | **Fix semantic recall return type bug** | `_decide()` treats `SemanticQuery` as a list. The semantic memory recall path is broken. | LOW |
| 1.7 | **Pass available_actions to advisory routing** | `_route_with_advisory()` passes `available_actions=None`, preventing the prefrontal tier from evaluating alternatives with the WorldModel. | LOW |

### Tier 2: High Impact (Wire existing systems together)

| # | What | Why | Effort |
|---|------|-----|--------|
| 2.1 | **Wire CuriosityDrive -> agent reward** | Intrinsic curiosity reward is computed but never added to the agent's actual reward. Curiosity-driven exploration is non-functional. | LOW |
| 2.2 | **Wire ValidatedImagination -> WorldModel training** | Imagination validation data is published to EventBus with zero subscribers. WorldModel never learns from its mistakes. | LOW |
| 2.3 | **Wire PatternDistiller into PatternConsolidator** | Distiller is injected but never called. Behavioral/state pattern extraction is dead code. | LOW |
| 2.4 | **Wire AwarenessPulse anomalies -> AutoHealer** | `holon.anomaly_detected` has no subscribers. Health anomalies are logged and forgotten. | LOW |
| 2.5 | **Wire PatternAdvisory -> EndocrineSystem** | High threat advisory should trigger adrenaline. High opportunity should trigger dopamine. Advisory publishes but Endocrine does not subscribe. | LOW |
| 2.6 | **Wire EndocrineSystem -> SignalPriorityResolver** | Hormone levels should modulate signal gain. Currently signal triage is static. | LOW |
| 2.7 | **Implement competitive ignition (GWT)** | Add ignition threshold to PatternCortex. Below threshold = quiet advisory. Above = full broadcast. Creates the two-state dynamic GWT requires. | MEDIUM |
| 2.8 | **Add top-down attentional gating (TRN analog)** | Cortex -> PatternBus feedback. Advisory controls what signals the bus processes next step. | MEDIUM |
| 2.9 | **Implement habituation/sensory adaptation** | Salience decay on repeated identical patterns. Prevents signal flooding. | LOW |
| 2.10 | **Wire self-awareness into agent step lifecycle** | Agents consult `holon_know_self()` during `_observe()` and `_decide()`. Closes the strange loop. | MEDIUM |

### Tier 3: Architectural (Deepen compliance)

| # | What | Why | Effort |
|---|------|-----|--------|
| 3.1 | **Make sensing fractal** | Define `SenseProtocol` with 3 detectors at every scale. Same generator, different resolution. | HIGH |
| 3.2 | **Make decision routing fractal** | 3-tier routing at tissue, organ, organism levels. Same cascade at every scale. | HIGH |
| 3.3 | **Implement COMPARE step (prediction error)** | Bridge between PREDICT and LEARN. Feeds CuriosityDrive, WorldModel, EndocrineSystem. | MEDIUM |
| 3.4 | **Implement goal management** | Explicit goals that persist across steps, drive attention and action. BDI architecture. | HIGH |
| 3.5 | **Add INHIBIT/go-no-go to DecisionRouter** | Ability to suppress action. Currently all tiers produce an action or fall through to default. | LOW |
| 3.6 | **Implement reconsolidation** | Retrieved memories become labile and can be updated. Currently ancestral memory is immutable. | MEDIUM |
| 3.7 | **Add dual-pathway habit system (Go/NoGo)** | Replace single habit lookup with competing pathways. Basal ganglia accuracy. | MEDIUM |
| 3.8 | **Meta-healing triad (who heals the healer)** | AutoHealer + HAVEN + SomaticMap form a healing triad where each monitors the other two. | MEDIUM |
| 3.9 | **Add DEVELOP step (apoptosis + re-differentiation)** | Structural change on slow timescale. Currently Mae can only grow, never prune. | HIGH |
| 3.10 | **Implement precision-weighted integration** | Weight each signal by its confidence. High-confidence sources dominate. | LOW |

---

## Key Files Reference

| File | Role | Audit Findings |
|------|------|----------------|
| `main.py` | 23-layer bootstrap, all wiring | Many wired but unconnected subsystems |
| `mae_core/agents/base_agent.py` | Step lifecycle | `_act()` is a two-line stub |
| `mae_core/agents/mycelial_agent.py` | Agent brain | Rich `_decide()`, broken `_learn()` path, no self-awareness in lifecycle |
| `mae_core/patterns/pattern_bus.py` | Thalamic relay | Signal mutation bug, no inhibitory gating, no priority draining |
| `mae_core/patterns/pattern_cortex.py` | Association cortex | No prediction, no competitive ignition |
| `mae_core/patterns/pattern_sense.py` | Cell membrane | Z-score self-inclusion bug, no habituation |
| `mae_core/patterns/pattern_consolidator.py` | Sleep consolidation | PatternDistiller never called |
| `mae_core/cognition/decision_router.py` | 3-tier cascade | Endocrine bias method missing, adrenaline re-check identical |
| `mae_core/cognition/world_model.py` | Predictive engine | Never called before observation |
| `mae_core/coordination/endocrine_system.py` | Hormonal system | Disconnected from pattern ecosystem and signal triage |
| `mae_core/memory/memory_consolidator.py` | Replay engine | Calls nonexistent agent methods |
| `mae_core/agents/mixins/episodic_memory.py` | Memory mixin | `_learn_from_batch()` updates no weights |
| `mae_core/emergent/auto_healer.py` | 3-phase healing | No step() method, reactive only |
| `mae_core/backbone/holon_protocol.py` | Self-awareness | know_* methods never called in step lifecycle |

---

## Individual Report Index

| # | Step | Report Location |
|---|------|----------------|
| 1 | SENSE | `data/audit-step-reports/01-sense.md` |
| 2 | ADVISE | `data/audit-step-reports/02-advise.md` |
| 3 | DECIDE | `data/audit-step-reports/03-decide.md` |
| 4 | ACT | `data/audit-step-reports/04-act.md` |
| 5 | LEARN | `data/audit-step-reports/05-learn.md` |
| 6 | CONSOLIDATE | `data/audit-step-reports/06-consolidate.md` |
| 7 | RECALL | `data/audit-step-reports/07-recall.md` |
| 8 | HEAL | `data/audit-step-reports/08-heal.md` |
| 9 | SELF-AWARENESS | `data/audit-step-reports/09-self-awareness.md` |
| 10 | HIDDEN STEPS | `data/audit-step-reports/10-hidden-steps.md` |

## Companion Documents

| Document | Location | Purpose |
|----------|----------|---------|
| Bugs and Fixes | `data/audit-bugs-and-fixes.md` | Every bug found, with file:line and fix |
| Mathematical Identity Compliance | `data/audit-mathematical-identity-compliance.md` | 10 holon capabilities compliance tables |
| Biological Comparison | `data/audit-biological-comparison.md` | All biological mechanisms vs Mae analogs |
| Upgrade Roadmap | `data/audit-upgrade-roadmap.md` | All recommendations deduplicated and ranked |
| External Research | `data/audit-external-research.md` | All papers, GitHub projects, URLs |
