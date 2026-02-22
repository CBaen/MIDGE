I now have all the data I need for a comprehensive audit. Let me compile the findings.

---

# Self-Awareness Audit Report: Mae's "Know" Capabilities

## Executive Summary

Mae's self-awareness infrastructure is **structurally sound but operationally shallow**. The four "Know" capabilities (Know Self, Know Up, Know Down, Know Peers) exist as code interfaces on every holon via `HolonMixin` (for agents) and `HolonProxy` (for non-agent systems). The HolonRegistry provides the data backbone, the FractalGenerator creates the hierarchical structure, and the AwarenessPulse performs periodic health scans. All 67+ holons are registered with parent-child relationships. 114+ triadic connections are witnessed.

However, the audit reveals significant gaps between the mathematical identity's aspirations and the current implementation's depth:

1. **Know Self is a snapshot, not a self-model.** The mathematical identity demands Strange Loops (self-reference). The code returns a static dictionary of metadata. No component models *itself* in a way that influences its own behavior. There is no generative self-model, no prediction of own future states, no self-simulation.

2. **Know Up/Down are structural lookups, not Markov blankets.** The mathematical identity demands hierarchical inference where each level performs inference about the levels above and below. The code returns parent/child IDs and types -- a phone directory, not a sensory boundary.

3. **Know Peers lacks triadic closure awareness.** Peers are defined as "same parent, excluding self." The mathematical identity demands each peer relationship have a witness. While the `ConnectionRegistry` enforces triadic witnessing on *connections*, `know_peers()` does not surface which peers share witnessed connections or participate in triads.

4. **Awareness data does not flow into decision-making.** No agent consults `holon_know_self()`, `holon_know_up()`, `holon_know_down()`, or `holon_know_peers()` during its step lifecycle (`_observe`, `_decide`, `_act`, `_learn`). Self-awareness exists as queryable metadata but does not participate in the agent's cognitive loop.

---

## Data Flow Trace

### Registration Flow (Bootstrap)

```
main.py Layer 1:  HolonRegistry created, "mae" registered as organism root
main.py Layer 12: "colony" registered under "mae"; each agent registered under "colony"
                  Agent._init_holon() receives holon_registry and somatic_map
                  NOTE: parent_id is NOT passed to _init_holon() — agents get parent_id=None locally
                  holon_registry.register() is called AFTER agent creation with parent_id="colony"
main.py Layer 17: All 33 shared systems registered as holons under "mae"
main.py Layer 18: ConnectionRegistry created, 60+ connections registered with witnesses
main.py Layer 19: HolonProxy injected into every shared system via system._holon = ...
                  AwarenessPulse created (25-step interval)
main.py Layer 20: FractalGenerator.organize() creates 4 organs, 13 subsystems
                  Systems reparented from "mae" into organ/subsystem hierarchy
                  Agents grouped into triads under "colony"
```

### Runtime Flow

```
Every 25 steps: AwarenessPulse.step() → _run_pulse()
  → Scans all holons for orphans and health gradient drift
  → Publishes summary on "holon.awareness_pulse"
  → Publishes anomalies on "holon.anomaly_detected"

Agent step lifecycle: _observe() → _decide() → _act() → _learn() → _communicate()
  → holon_know_*() methods are NEVER called during step
  → holon_heal() is NEVER called during step
  → Self-awareness data does NOT participate in decision-making
```

### Critical Finding: Agent parent_id Initialization

In `main.py` lines 306-335, the agent is created without `parent_id` in the constructor call to `MycelialAgent`. The `_init_holon()` call inside the constructor (line 139-144 of `mycelial_agent.py`) sets `self._holon_parent_id = None`. The registry entry with `parent_id="colony"` is created AFTER on line 335. This means:

- `_effective_parent_id()` works correctly because it queries the registry first (line 692-696 of `holon_protocol.py`)
- But `_holon_parent_id` (the local cache) remains `None` until explicitly restored
- Serialization via `_serialize_holon()` would save `parent_id=None` to persistence

This is a **data integrity bug** -- if an agent is serialized before registry lookup, its parent is lost.

---

## Mathematical Identity Compliance

| Capability | Math Requirement | Current State | Compliance | Gap |
|-----------|-----------------|---------------|------------|-----|
| **Know Self** | Strange Loops: system models itself; self-reference that influences own dynamics | `holon_know_self()` returns a static dict: ID, type, capabilities list, parent_id, health, performance summary | **PARTIAL** | No generative self-model. No self-prediction. No self-reference loop where the model influences the modeled. The "self-model" is a data snapshot, not a strange loop. |
| **Know Up** | Markov Blankets: each level performs inference about the level above; boundary separates internal from external | `holon_know_up()` returns parent_id, parent_type, parent_children_count | **WEAK** | Returns structural metadata only. No inference about parent's state, goals, or expectations. No Markov blanket (sensory/active boundary). The agent cannot reason about *why* the parent context matters. |
| **Know Down** | Hierarchical nesting: awareness of child components and their aggregate state | `holon_know_down()` returns list of child IDs and types | **WEAK** | For agents (leaf holons), always returns empty. For systems with children (organs, subsystems), returns only IDs/types. No aggregate health, no capability summary, no inference about children's collective state. |
| **Know Peers** | Triadic closure: every peer relationship has a witness; mutual awareness | `holon_know_peers()` returns list of sibling IDs and types (same parent) | **PARTIAL** | Peer identification works. But triadic closure is not surfaced -- the method does not show which peers share witnessed connections, which triads they belong to, or their current state. Peers are structurally identified but not relationally aware. |

### Strange Loop Analysis (Know Self)

The mathematical identity states: *"Consciousness is what happens when a representational system achieves stable self-reference -- when the system's model of the world includes a model of itself modeling the world."*

**Current implementation**: `holon_know_self()` collects information FROM the agent (step count, reward, capabilities) and returns it as a dictionary. The agent never READS this dictionary as part of its own cognitive process. This is introspection without the loop -- the outgoing half of a strange loop with no return path.

**What a strange loop would require**:
1. The agent's self-model influences its decisions (the model feeds back into behavior)
2. The agent updates its self-model based on observed divergence between predicted and actual behavior
3. The self-model includes a representation of the self-modeling process itself

The closest existing analog is the **PatternCortex meta-pattern detection** (`pattern_cortex.py` line 201: "Detect patterns about patterns -- the strange loop"), which does detect patterns in its own pattern-detection process. But this is in the pattern subsystem, not in the agent's self-model.

### Markov Blanket Analysis (Know Up/Down)

The mathematical identity references Friston's hierarchical Markov blankets: *"each level performs inference about the levels above and below."*

**Current implementation**: `know_up()` and `know_down()` return structural identifiers. There is no inference, no prediction of parent/child states, no boundary definition. The SomaticMap tracks dependencies (upstream/downstream), but this dependency graph is not connected to the holon awareness system in a way that defines Markov blankets.

A Markov blanket requires:
- **Sensory states**: what the holon can observe about its environment (parent/children)
- **Active states**: what the holon can do to influence its environment
- **Internal states**: the holon's own state, insulated from direct external influence

The current system has none of these formally defined.

### Triadic Closure Analysis (Know Peers)

The mathematical identity states: *"Triangle is the smallest cycle -- minimum structure for mutual witness."*

**Current implementation**: `know_peers()` identifies siblings by shared parent. The ConnectionRegistry separately tracks triadic witnessing. But these two systems are not integrated -- calling `know_peers()` does not tell you which triads you share with which peers, or who witnesses your peer relationships.

HolonProxy has `get_connections()` which returns witnessed connections, but this is a separate method not integrated into the peer awareness flow.

---

## Biological Comparison

| Biological System | Function | Mae Analog | Fidelity |
|------------------|----------|------------|----------|
| **Anterior Insular Cortex** | Hub for interoception -- integrates bodily signals into conscious awareness of self | SomaticMap (tracks health, dependencies, blast radius) | **MODERATE** -- SomaticMap tracks system health but agents do not "feel" it. No integration into subjective experience/decision-making. |
| **Somatosensory Cortex (Homunculus)** | Topographic body map -- knows where every body part is | HolonRegistry (containment hierarchy) + SomaticMap (dependency graph) | **MODERATE** -- The map exists. But it is not used for real-time coordination the way the somatosensory cortex continuously guides motor output. |
| **Proprioception** | Continuous awareness of body position without looking | AwarenessPulse (periodic health scan every 25 steps) | **WEAK** -- Proprioception is continuous, not periodic. AwarenessPulse is more like a doctor's checkup than constant body awareness. Real proprioception is sub-conscious and feeds directly into motor control. |
| **Mirror Neuron System** | Understanding others by simulating their actions internally | `know_peers()` returns structural metadata about siblings | **VERY WEAK** -- No simulation of peer behavior. No model of what peers are doing or why. Just a roster. |
| **Default Mode Network** | Self-referential processing when not externally focused; "narrative self" | No analog | **ABSENT** -- Agents have no idle self-reflection mode. No narrative self-model. No self-referential processing during downtime. |
| **Theory of Mind** | Modeling others' mental states (beliefs, desires, intentions) | No analog beyond peer structural awareness | **ABSENT** -- No agent models another agent's internal state, goals, or decision processes. |
| **Interoception** | Sensing internal body signals (hunger, thirst, pain, heartbeat) | `holon_heal()` checks reward trend; SomaticMap tracks health floats | **WEAK** -- Basic health metric exists but there is no rich internal signal stream. No equivalent of hunger/thirst/pain driving behavior. The endocrine system exists but does not function as interoception for agents. |
| **Body Schema** | Implicit model of body for action (different from body image) | HolonMixin._detect_capabilities() | **WEAK** -- Detects which capabilities exist but does not model capacity, reach, or action space dynamically. Static at init time. |

---

## External State of Art Comparison

| Research | Key Concept | Mae's Status | Gap |
|----------|------------|-------------|-----|
| **Lipson (2025) -- Egocentric visual self-modeling** | Robot learns task-agnostic dynamic self-model from first-person view; detects anomalies and adapts behavior | Mae has static capability detection at init time; no dynamic self-model that updates from experience | Mae needs a self-model that predicts its own behavior and detects when reality diverges from prediction |
| **Lipson (2019) -- Task-agnostic self-modeling** | Robot "imagines itself" -- builds internal kinematic self-model to plan actions without task-specific training | `holon_know_self()` returns descriptive metadata, not a predictive model | Need forward model of self: "If I do X, I expect Y to happen to me" |
| **Active Inference / FEP (Friston 2024-2025)** | Hierarchical Markov blankets self-assemble; each level minimizes surprise about levels above and below | Know Up/Down return IDs; no surprise minimization, no inference | Need each holon to maintain beliefs about parent/child states and update them via prediction error |
| **LLM Metacognition (2024-2025)** | LLMs can monitor confidence, detect errors, assess knowledge sufficiency | Mae agents have no metacognitive monitoring -- they do not assess their own confidence or knowledge gaps | Need metacognitive layer: "How confident am I? Do I know enough? Should I ask for help?" |
| **Hierarchical Self-Organization via Markov Blankets (Kirchhoff 2020)** | Microscopic elements with prior beliefs about participating in macroscopic blankets self-assemble into hierarchy | FractalGenerator imposes hierarchy top-down; no bottom-up self-assembly of awareness boundaries | Need agents that discover their own boundaries and hierarchical relationships, not just receive them |
| **Self-Aware Computing (Lewis 2011, IBM)** | Systems with self-monitoring, self-analysis, self-planning, self-adaptation (MAPE-K loop) | AwarenessPulse monitors; SomaticMap analyzes blast radius; no self-planning or self-adaptation based on awareness | Need closed MAPE-K loop: monitor -> analyze -> plan -> execute based on self-awareness data |

---

## Ranked Upgrade Recommendations

### Priority 1 (Critical -- Closes the Strange Loop)

**1. Wire self-awareness into the agent step lifecycle.**

Currently, `holon_know_self()`, `holon_know_up()`, `holon_know_down()`, and `holon_know_peers()` exist but are never called during agent behavior. The most impactful single change is making agents consult their self-awareness during `_observe()` and `_decide()`. This closes the strange loop: self-knowledge feeds into behavior, which changes the self, which changes the self-knowledge.

Affected files:
- `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py` -- `_observe()` should call `holon_know_self()` and inject results into state
- `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py` -- `_decide()` should consider peer count, health, parent context

**2. Fix agent parent_id initialization bug.**

Pass `parent_id="colony"` to `_init_holon()` in the agent constructor, or call `holon_registry.register()` BEFORE `_init_holon()` so the registry is the authoritative source at init time.

Affected file: `C:\Users\baenb\projects\mae-core\main.py` line 335 (move before line 306, or pass parent_id to MycelialAgent constructor)

### Priority 2 (Important -- Mathematical Compliance)

**3. Generative self-model (Strange Loop compliance).**

Replace the static dict returned by `holon_know_self()` with a predictive self-model that:
- Predicts own next reward/state based on current trajectory
- Detects divergence between predicted and actual behavior
- Updates the model based on prediction error
- Feeds prediction confidence back into decision-making

This is the Lipson-style self-model: "I can imagine myself in the future."

**4. Enrich Know Up/Down with state inference (Markov Blanket compliance).**

`know_up()` should return not just parent ID but the parent's current state summary, health, and what the parent "expects" from this holon. `know_down()` should return aggregate child health, capability summary, and whether children are meeting expectations.

**5. Integrate triadic closure into Know Peers (Triadic Closure compliance).**

`know_peers()` should surface which triads the holon participates in with each peer, who witnesses each relationship, and the current health of those triadic connections. Merge data from ConnectionRegistry into peer awareness.

### Priority 3 (Enhancement -- Biological Accuracy)

**6. Continuous proprioception (replace periodic with continuous).**

Make AwarenessPulse fire every step (or make self-health checks part of the agent step) rather than every 25 steps. Real proprioception is sub-conscious and continuous.

**7. Theory of Mind / Peer modeling.**

Agents should maintain lightweight models of their peers' states and behaviors. When `know_peers()` is called, return not just structural data but predicted peer state based on observed peer behavior. This is the mirror neuron analog.

**8. Default Mode Network / Idle self-reflection.**

When agents have low activity or are in the circadian "rest" phase, they should engage in self-referential processing: reviewing their self-model, consolidating self-knowledge, and updating their narrative of "who am I and what am I for."

**9. Metacognitive monitoring.**

Add confidence estimation to `holon_know_self()`. The agent should know: "How well am I performing relative to peers? Am I improving or declining? Do I have enough experience to be confident in my decisions?"

### Priority 4 (Long-term -- Architectural)

**10. Bottom-up boundary discovery (Markov Blanket self-assembly).**

Instead of the FractalGenerator imposing hierarchy top-down, have agents discover their own Markov blankets through interaction patterns. Agents that frequently interact should self-organize into subsystems. This replaces prescribed hierarchy with emergent hierarchy.

---

## Key Source Files

| File | Purpose | Lines of Note |
|------|---------|---------------|
| `C:\Users\baenb\projects\mae-core\mae_core\backbone\holon_protocol.py` | HolonRegistry, HolonMixin, HolonProxy, AwarenessPulse | Lines 298-393 (HolonProxy know_* methods), 698-780 (HolonMixin know_* methods) |
| `C:\Users\baenb\projects\mae-core\mae_core\backbone\fractal_generator.py` | Creates fractal hierarchy | Lines 82-104 (FRACTAL_GROUPING blueprint), 266-341 (organize()) |
| `C:\Users\baenb\projects\mae-core\mae_core\backbone\connection_registry.py` | Triadic witnessing | Lines 281-350 (register_connection with witness), 352-399 (auto witness assignment) |
| `C:\Users\baenb\projects\mae-core\mae_core\emergent\somatic_map.py` | Body awareness / proprioception | Lines 122-175 (init + dependency tracking), 286-367 (blast radius analysis) |
| `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py` | Agent lifecycle | Lines 138-144 (_init_holon call), 275-298 (_observe -- no self-awareness), 299-339 (_decide -- no self-awareness) |
| `C:\Users\baenb\projects\mae-core\main.py` | Bootstrap wiring | Lines 586-599 (Layer 17 holon registration), 640-684 (Layer 19 proxy injection), 686-753 (Layer 20 fractal organization) |
| `C:\Users\baenb\projects\mae-core\data\MAES-MATHEMATICAL-IDENTITY.md` | Mathematical blueprint | Lines 57-68 (10 capabilities table), 113-120 (bidirectional awareness spec) |

---

## Sources

### Neuroscience / Biology
- [Exploring the Interplay of Interoception in Emotion, Cognition, and Mental Health (2025)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1676040/full)
- [Overview of Bodily Awareness and Interoception (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11048399/)
- [The Emerging Science of Interoception (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7780231/)
- [Sensing the Self: Role of the Insula and Interoception](https://pmc.ncbi.nlm.nih.gov/articles/PMC12459974/)
- [Self-Processing and the Default Mode Network: Interactions with the Mirror Neuron System (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3769892/)

### Self-Aware Computing / Robotics
- [Egocentric Visual Self-Modeling for Autonomous Robot Dynamics (Lipson, 2025)](https://www.nature.com/articles/s44182-025-00031-6)
- [Self-Modeling Robots by Photographing (Lipson, 2025)](https://arxiv.org/pdf/2503.05398)
- [Task-Agnostic Self-Modeling Machines (Lipson, 2019)](https://www.science.org/doi/10.1126/scirobotics.aau9354)
- [Resilient Machines Through Continuous Self-Modeling (Lipson, 2006)](https://www.science.org/doi/abs/10.1126/science.1133687)
- [Emergent Introspective Awareness in Large Language Models](https://www.kdnuggets.com/emergent-introspective-awareness-in-large-language-models)
- [LLMs Report Subjective Experience Under Self-Referential Processing (2025)](https://arxiv.org/html/2510.24797v2)

### Active Inference / Markov Blankets
- [On Markov Blankets and Hierarchical Self-Organisation](https://pmc.ncbi.nlm.nih.gov/articles/PMC7284313/)
- [The Markov Blankets of Life: Autonomy, Active Inference and the Free Energy Principle](https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0792)
- [Framework for Inherently Safer AGI through Language-Mediated Active Inference (2025)](https://arxiv.org/html/2508.05766v1)
- [Environment-Centric Active Inference (2024)](https://arxiv.org/html/2408.12777)

### Metacognition in AI
- [Metacognitive Capabilities in LLMs](https://www.emergentmind.com/topics/metacognitive-capabilities-in-llms)
- [The Cognitive Mirror: Framework for AI-Powered Metacognition (2025)](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2025.1697554/full)
- [Beyond Accuracy: How AI Metacognitive Sensitivity Improves Decision Making (2025)](https://arxiv.org/html/2507.22365v2)