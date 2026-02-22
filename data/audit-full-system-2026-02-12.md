# Mae Full System Audit — 2026-02-12

**Audit Type:** Comprehensive 6-Team System Audit
**Audit Date:** 2026-02-12
**Prior Audit Health:** 5.2/10
**Current Health:** 7.8/10
**Systems Audited:** 73
**Agents Deployed:** ~25 (6 primary + sub-agents)

---

## Executive Summary

Mae's overall system health has improved from 5.2/10 to **7.8/10** since the last audit cycle. Six audit teams were deployed, each spawning 3-4 sub-agents, for a total of approximately 25 agents examining the codebase in parallel. Coverage included all 73 systems, 29 bootstrap layers, 107 EventBus channels, and all mixin interfaces.

**Key findings:**
- The fractal triadic architecture is structurally sound and well-implemented
- 19 new biological systems were added at high quality (Grade: A-)
- The primary remaining issues are **wiring contact failures masked by `except Exception` blocks**
- 12 bugs identified (3 critical, 1 high, 5 medium, 2 info, 1 moderate)
- Fixing the critical/high bugs alone would bring health to approximately 8.5/10

---

## Team Reports

### Team Alpha: Signal Paths (Grade: C+)

**Scope:** All EventBus channels, publisher-subscriber mapping, end-to-end signal delivery.

| Metric | Value |
|--------|-------|
| Distinct EventBus channels | 107 |
| Verified end-to-end | 54 |
| Orphan channels (published, no subscriber) | 42 |
| Dead subscriptions (subscriber, no publisher) | 2 |

**5 Critical Broken Signal Chains:**

| # | Broken Chain | Details |
|---|-------------|---------|
| 1 | `healing.started` / `healing.complete` / `healing.failed` | `healing.started` never published; `healing.complete` and `healing.failed` have no subscribers |
| 2 | `temporal.causal_link_discovered` | Never published, but `CausalReasoningEngine` and `CausalEngineTranslator` subscribe to it |
| 3 | `learning.policy_success` | Never published, but `KnowledgeBase` subscribes |
| 4 | `cognition.model_trained` | Published but nobody listens |
| 5 | `cognition.collective_dream_complete` | Published but never consumed |

**Assessment:** Over 40% of EventBus channels are orphaned — signals are being published into the void. The 54 verified channels demonstrate the architecture works; the gap is incomplete wiring during system integration.

---

### Team Beta: Mathematical Identity (Grade: C+)

**Scope:** Compliance with Mae's 8 Mathematical Laws as defined in `data/MAES-MATHEMATICAL-IDENTITY.md`.

| Law | Name | Compliance | Notes |
|-----|------|------------|-------|
| 1 | No Bare Dyads | **COMPLIANT** | 132 connections, 0 bare dyads |
| 2 | Triadic Generator | **MOSTLY COMPLIANT** | 21/24 clean triads, 3 advisory-only |
| 3 | Holon Protocol | **MOSTLY COMPLIANT** | 10/10 on agents, 5/10 on proxies, 94 holons total |
| 4 | Fractal Self-Similarity | **PARTIAL** | Topology correct, behaviors differ at scales |
| 5 | Stem Cell Principle | **COMPLIANT** | 20 genes, 7 roles, redifferentiation works |
| 6 | Autopoietic Closure | **PARTIAL** | Architecture correct, 3 runtime bugs break closure loops |
| 7 | Rule of 3/5 | **COMPLIANT** | 16/16 processes, 62 validators, 100% coverage |
| 8 | Eight Properties of Consciousness | **PARTIAL** | 3/8 fully present, 5/8 partial |

**Assessment:** The mathematical identity is well-embedded in the architecture. Full compliance on Laws 1, 5, and 7. The partial compliance on Laws 4, 6, and 8 reflects runtime/behavioral gaps rather than architectural ones — the structure is right, but some behaviors are not yet implemented at every fractal scale.

---

### Team Gamma: API Contracts (Grade: C)

**Scope:** Interface signatures, return types, parameter names, step hook contracts across all 73 systems.

**Critical Bugs (3):**

| Bug | Location | Impact |
|-----|----------|--------|
| Layer 15 EventBus callbacks take 1 argument, bus sends 2 | `main.py` | ALL cross-system wiring dead |
| `collective_plan` returns tuple, code expects dict | `mycelial_agent.py` | Dream planning never produces action |
| `broadcast_intention` passes `position=` not `target_position=` | `mycelial_agent.py` | Predictive field broadcasting dead |

**Moderate Bugs (2):**

| Bug | Location | Impact |
|-----|----------|--------|
| `_signal_bus` getattr checks wrong attribute name | `mycelial_agent.py` | Morphogenesis gap signal dead |
| `WorldlinePlanner` entity_id type mismatch | `worldline_planner.py` | Potential type errors |

**Systemic Issues:**
- **14 step hooks** receive `current_step=0` due to missing lambda wrappers — time-awareness broken across the organism
- **16 bare `except Exception` blocks** silently swallow errors, masking bugs that would otherwise surface immediately
- **65+ interfaces** verified correct (the majority of the API surface is sound)

**Assessment:** The API surface is mostly correct, but the 3 critical bugs disable major subsystems entirely. The `except Exception` pattern is the single most damaging practice in the codebase — it converts hard failures into silent degradation.

---

### Team Delta: 19 New Biological Systems (Grade: A-)

**Scope:** All 19 newly-added biological systems — code quality, test coverage, biological accuracy, integration.

| Metric | Value |
|--------|-------|
| Systems audited | 19 |
| Systems with meaningful `step()` | 19/19 |
| Systems with EventBus publishing | 19/19 |
| Systems with tests | 19/19 |
| Systems with `serialize`/`restore` | 19/19 |
| Total test methods | ~260+ |
| Test files | 7 |
| Biology inaccuracies | 0 |

**Bugs Found (3):**

| Bug | System | Severity | Details |
|-----|--------|----------|---------|
| Energy normalization error | `OrganismState` | HIGH | Energy range 0-100 treated as 0-1; energy always reads 1.0 |
| Wrong EventBus channel | `CirculatorySystem` | MEDIUM | Subscribes to incorrect channel; starvation response dead |
| Stub method | `ReproductiveSystem` | LOW | `_select_spawn_role` is a stub, returns hardcoded value |

**Assessment:** The 19 new biological systems represent the highest-quality addition to Mae in any audit cycle. Code is consistent, well-tested, biologically accurate, and follows all project conventions. The 3 bugs are isolated and fixable.

---

### Team Epsilon: Bug Hunt & Dead Code (Grade: B)

**Scope:** Silent failures, dead code paths, exception handling, regression verification.

**Silent Failures (5, medium severity):**

| # | Failure | Impact |
|---|---------|--------|
| 1 | `tick_episodic_memory()` never called | Reconsolidation and activation decay never tick |
| 2 | `decay_working_memory()` called on wrong object | Working memory never decays |
| 3 | `get_consolidation_statistics()` missing from interface | Statistics reporting incomplete |
| 4 | `get_local_reward()` doesn't exist on model | VDN reward distribution broken |
| 5 | Reward stats drift over time | Cumulative floating-point drift in statistics |

**Dead Code Paths (3):**

| # | Dead Code | Location |
|---|-----------|----------|
| 1 | `tick_episodic_memory` method | Defined but never invoked |
| 2 | VDN distribution logic | References nonexistent `get_local_reward` |
| 3 | Old signal handler setup | Superseded by new wiring |

**Other Findings:**
- **23 silent `except: pass` blocks** across the codebase
- **9/10 previously fixed bugs** confirmed still fixed (1 regression-free)
- **0 TODO/FIXME comments** in the codebase

**Assessment:** The codebase is clean of technical debt markers (no TODOs), and previous fixes have held. The silent failures are all caused by the same root pattern: methods that exist but are never called, or are called on the wrong object, with exceptions masking the error.

---

### Team Zeta: Roadmap & Re-Grade (Health: 7.8/10)

**Scope:** Roadmap progress tracking, lifecycle capability re-grading, overall health assessment.

**Roadmap Progress:**

| Tier | Completed | Partial | Remaining | Completion |
|------|-----------|---------|-----------|------------|
| Tier 1 (Foundation) | 9/9 | 0 | 0 | **100%** |
| Tier 2 (Integration) | 14/15 | 0 | 1 | **93%** |
| Tier 3 (Emergence) | 5/20 | 4 | 11 | **35%** |
| **Total** | **28/44** | **4** | **12** | **68%** |

**Lifecycle Capability Re-Grades:**

| Capability | Previous Grade | Current Grade | Change |
|------------|---------------|---------------|--------|
| PREDICT | F | B+ | +5 levels |
| ACT | F | B+ | +5 levels |
| SELF-AWARENESS | D | B | +3 levels |
| SENSE | C+ | B+ | +2 levels |
| LEARN | B- | A- | +2 levels |
| RECALL | C+ | B+ | +2 levels |
| ADVISE | B- | A- | +2 levels |
| DECIDE | B | A- | +1 level |
| CONSOLIDATE | B | A- | +1 level |
| HEAL | B | A- | +1 level |

**Assessment:** Every lifecycle capability improved. The most dramatic improvements were in PREDICT and ACT (F to B+), reflecting the new biological systems' contributions to temporal reasoning and motor coordination. Tier 1 is fully complete; Tier 3 (emergence behaviors) is the frontier.

---

## Critical Bugs — Prioritized Fix List

| Priority | Bug | File | Severity | Impact | Effort |
|----------|-----|------|----------|--------|--------|
| 1 | 11 Layer 15 EventBus callbacks take 1 arg, bus sends 2 | `main.py` | **CRITICAL** | ALL cross-system wiring dead | LOW — add `*args` or fix lambda signatures |
| 2 | `collective_plan` returns tuple, code expects dict | `mycelial_agent.py` | **CRITICAL** | Dream planning never produces action | LOW — match return type |
| 3 | `broadcast_intention` passes `position=` not `target_position=` | `mycelial_agent.py` | **CRITICAL** | Predictive field broadcasting dead | LOW — rename kwarg |
| 4 | `_signal_bus` getattr checks wrong name | `mycelial_agent.py` | **MODERATE** | Morphogenesis gap signal dead | LOW — fix attribute name |
| 5 | `OrganismState` energy: 0-100 treated as 0-1 | `organism_state.py` | **HIGH** | Energy always reads 1.0 | LOW — normalize correctly |
| 6 | `CirculatorySystem` subscribes to wrong channel | `circulatory_system.py` | **MEDIUM** | Starvation response dead | LOW — fix channel name |
| 7 | `tick_episodic_memory()` never called | `mycelial_agent.py` | **MEDIUM** | Reconsolidation/activation never tick | LOW — add call to step |
| 8 | `decay_working_memory()` on wrong object | `mycelial_agent.py` | **MEDIUM** | Working memory never decays | LOW — fix target object |
| 9 | `get_local_reward` doesn't exist | `model.py` | **MEDIUM** | VDN distribution broken | MEDIUM — implement method |
| 10 | 14 step hooks receive `current_step=0` | `main.py` | **MEDIUM** | Time-awareness broken | LOW — wrap in lambdas |
| 11 | 42 orphan EventBus channels | various | **INFO** | 40% signals undelivered | MEDIUM — wire subscribers |
| 12 | `agent_factory` lambda broken | `main.py` | **MEDIUM** | Can't spawn new agents at runtime | LOW — fix closure |

**Note:** Bugs 1-3 are the highest-impact, lowest-effort fixes. Resolving them would restore cross-system wiring, dream planning, and predictive broadcasting in a single pass.

---

## Remaining Roadmap Gaps — Top 10

| Priority | Gap | Effort | Impact |
|----------|-----|--------|--------|
| 1 | No trainable policy/value network (LEARN computes TD errors, updates nothing) | HIGH | Core learning loop incomplete |
| 2 | No goal management / BDI architecture | HIGH | No persistent intentions or desires |
| 3 | Two consolidation systems uncoordinated | MEDIUM | Redundant/conflicting memory consolidation |
| 4 | No Go/No-Go inhibition in DecisionRouter | LOW | Decisions cannot be suppressed |
| 5 | Fractal sensing/deciding not implemented (Law 4) | HIGH | Self-similarity breaks at behavior level |
| 6 | Precision-weighted integration missing in ADVISE | LOW | All advice weighted equally |
| 7 | No immune memory / adaptive healing | MEDIUM | Healing doesn't learn from past repairs |
| 8 | Multi-step prefrontal rollouts not used | LOW | Planning limited to single step |
| 9 | Generative self-model incomplete (strange loop) | HIGH | Self-awareness lacks self-modeling |
| 10 | Sequence-aware replay absent | MEDIUM | Memory replay ignores temporal order |

---

## New Gaps from Biological Systems Integration

These gaps were discovered during the audit of the 19 new biological systems and are **not tracked in the original roadmap**.

| # | Gap | Risk Level | Notes |
|---|-----|-----------|-------|
| 1 | `OrganismState` reflex thresholds are uncalibrated estimates | MEDIUM | Thresholds chosen without empirical tuning |
| 2 | 18 biological system signal paths not fully audited pre-integration | MEDIUM | Only 1 of 19 systems had signal paths traced end-to-end before integration |
| 3 | `OrganismState` is single point of integration failure (no redundancy) | HIGH | If `OrganismState` fails, all biological systems lose coordination |
| 4 | Emotional valence has no reset/decay if `EmotionalSystem` stops publishing | MEDIUM | Stale emotional state persists indefinitely |
| 5 | Cross-system feedback loops may not close end-to-end | MEDIUM | Circular dependencies assumed but not verified at runtime |
| 6 | Dormant system activation depth unmeasured | LOW | Unknown how deep dormant systems go before activating |
| 7 | `EndocrineSystem` to `SignalPriorityResolver` unwired | LOW | Only Tier 2 gap from biological integration |

---

## Systemic Patterns

Three patterns account for the majority of all findings:

1. **`except Exception` masking:** 16+ bare exception blocks silently swallow errors that would otherwise reveal broken wiring, wrong argument counts, and missing methods. This is the single largest source of hidden bugs.

2. **Orphan signals:** 42 of 107 EventBus channels have no subscribers. Systems publish events that nobody consumes. The architecture is correct — the wiring is incomplete.

3. **Lambda closure bugs:** Step hooks, factory functions, and callbacks share a common pattern of capturing loop variables or passing wrong argument counts. These are low-effort fixes with high impact.

---

## Methodology

| Aspect | Detail |
|--------|--------|
| Audit teams | 6 primary agents (Alpha through Zeta) |
| Sub-agents per team | 3-4 |
| Total agents | ~25 |
| Coverage | All 73 systems, 29 bootstrap layers, 107 EventBus channels, all mixin interfaces |
| Runtime verification | Bootstrapped Mae with 3 agents, 10 steps |
| Cross-reference | All findings cross-referenced against previous 10-agent audit |
| Code analysis | Static analysis of all source files in `mae_core/` and `main.py` |
| Test verification | Full test suite execution (`python -m pytest tests/ -v`) |

---

## Conclusion

Mae's architecture has matured dramatically, rising from 5.2/10 to **7.8/10** since the prior audit. The fractal triadic structure is mathematically sound. The 19 new biological systems are the highest-quality addition to the codebase to date (Grade: A-). Every lifecycle capability improved, with PREDICT and ACT making the largest jumps (F to B+).

The remaining issues fall into two categories:

1. **Wiring bugs (fixable now):** The 12 bugs listed above are primarily contact failures — wrong argument counts, wrong parameter names, missing method calls. Most are LOW effort. Fixing them would bring health to approximately **8.5/10**.

2. **Architectural gaps (future work):** Goal management, trainable networks, fractal behaviors at all scales, and generative self-modeling represent the path from 8.5 to 10. These are design challenges, not bugs.

**Recommended next action:** Fix bugs 1-3 (all CRITICAL, all LOW effort) to restore cross-system wiring, dream planning, and predictive broadcasting. Then address the `except Exception` blocks to prevent future bugs from being masked.

---

*Report generated by 6-team audit, ~25 agents, 2026-02-12.*
*Prior audit baseline: 10-agent audit, health 5.2/10.*
