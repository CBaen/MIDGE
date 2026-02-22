> Generated from 10-agent audit conducted 2026-02-11. ~50 sub-agents. Sources: biology papers, GitHub, research papers, full codebase trace.

# ACT STEP AUDIT REPORT: Mae's Action Execution

## Executive Summary

**The ACT step in Mae is the weakest link in the entire lifecycle.** The `_act()` method in `BaseAgent` is a two-line stub that stores the action and returns 0.0. Neither `MycelialAgent` nor any mixin overrides it. No action in Mae's current implementation changes any environmental state, moves an agent in space, alters the model, affects another agent directly, or produces any observable effect in the world. The decision made in `_decide()` is discarded — its return value is passed to `_act()` which stores it as `self.last_action` and returns a constant zero reward.

Actions do not cause effects. The causal chain is broken at the most critical junction.

**Key findings:**

1. `_act()` is never overridden — it is the only lifecycle method that remains a pure stub
2. There is no environment to act upon — no grid, no continuous space, no task queue
3. Reward is always 0.0 from `_act()`, meaning all learning signal comes from external VDN distribution, not from action consequences
4. Stigmergy deposits happen in `_learn()` and `_communicate()`, not in `_act()` — they are post-hoc annotations, not action effects
5. No efference copy mechanism exists — the system cannot predict its own action consequences
6. ACT is not triadic and does not exist fractally at every scale

---

## Data Flow Trace: Decision to Environmental Effect

### Current Flow

```
BaseAgent.step():
  1. _observe()    -> builds state vector, senses stigmergy markers
  2. _decide()     -> returns action (int, dict, or None)
  3. _act(action)  -> stores self.last_action, returns 0.0
  4. _learn(action, reward=0.0) -> stores experience, deposits markers
  5. _communicate() -> processes GNN messages, deposits exploration trail
```

### Critical Files

| File | Role | Finding |
|------|------|---------|
| `C:\Users\baenb\projects\mae-core\mae_core\agents\base_agent.py` (lines 92-95) | `_act()` stub | Stores action, returns 0.0. Never overridden. |
| `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py` (lines 299-339) | `_decide()` override | Rich cascade: advisory -> memory -> world model -> default. But output goes nowhere. |
| `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py` (lines 394-421) | `_learn()` override | Deposits stigmergy markers, stores experience. Uses reward from `_act()` which is always 0.0. |
| `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py` (lines 423-433) | `_communicate()` override | Processes GNN messages, deposits exploration trail, shares patterns. |
| `C:\Users\baenb\projects\mae-core\mae_core\model.py` (line 118) | Model step | `self.agents.shuffle_do("step")` — calls each agent's `step()`. No environment dynamics. |

### The Gap

The action returned by `_decide()` is an integer (from world model), a dict (from decision router), or the return of `_select_action()` which returns 0. This value is:
- Passed to `_act()` which stores it and returns 0.0
- Passed to `_learn()` which uses the 0.0 reward to store an experience

**Nothing in the environment changes as a result of the action.** The "action" is a label attached to a step, not an effect.

---

## Mathematical Identity Compliance

| Criterion | Required By | Status | Detail |
|-----------|------------|--------|--------|
| **Cause-effect power** | IIT Postulate 1 | FAIL | `_act()` has no effect. It takes no cause from the environment and makes no difference. Reward is constant 0.0. Per IIT 4.0: "to exist physically is to take and make a difference." ACT takes no difference and makes none. |
| **Triadic structure** | Mae's generator principle | FAIL | ACT is a single method (`_act()`). It is not composed of three complementary components. Compare: DECIDE has three tiers (reflex/habit/prefrontal). OBSERVE has stigmergy + state vector + advisory. ACT has one stub. |
| **Fractal (every scale)** | Holarchy requirement | FAIL | `BaseAgent._act()` exists at cell level. `OctopusAgent.submit_task()` exists at tissue level. `MorphogenesisCoordinator.handle_novel_problem()` exists at organ level. But these are not connected by a common ACT protocol. The HolonMixin defines `holon_act()` but it just delegates to `_act()` which is the stub. |
| **Intrinsicality** | IIT Postulate | PARTIAL | The action is stored on `self.last_action`, giving the agent internal awareness of what it "did." But without external effect, this is a record of intention, not action. |
| **Information** | IIT Postulate | FAIL | All actions produce the same result (reward 0.0). There is no specificity — different actions have identical consequences. |
| **Integration** | IIT Postulate | FAIL | ACT is not integrated with the rest of the lifecycle in a causal way. DECIDE outputs are not consumed by ACT in any meaningful way. ACT outputs (reward) do not vary based on input. |

---

## Biological Comparison

| Biological System | What Biology Does | What Mae Does | Gap |
|---|---|---|---|
| **Motor cortex (M1)** | Generates specific motor commands: activates particular muscle groups with precise timing. Commands are specific — reaching activates different neurons than grasping. | `_act()` stores an action label and returns 0.0. No specific execution for different action types. | No motor specificity. All "actions" are identical from the environment's perspective. |
| **Efference copy** | Motor cortex sends a copy of the motor command to cerebellum. Cerebellum uses this to predict sensory consequences of the action BEFORE they happen. Mismatch drives motor learning. | No efference copy. The world model can predict but is never asked to compare predictions against actual outcomes of actions. | Missing entirely. This is the core mechanism by which biology validates its actions in real-time. |
| **Neuromuscular junction** | The interface where neural commands become physical force. Lower motor neurons release acetylcholine at the NMJ, causing muscle contraction. There is a discrete, traceable moment where signal becomes force. | No equivalent. There is no interface where a decision becomes an environmental change. | The entire "effector" stage is absent. |
| **Cerebellum coordination** | Timing, error correction, smooth multi-step sequences. Receives efference copy during planning, generates predictive sensory model, corrects errors during execution. | The DecisionRouter has three tiers (analogous to reflex/habit/cortical processing) but this is in DECIDE, not ACT. No cerebellar equivalent adjusts actions during or after execution. | The cerebellum analog should sit in ACT, not DECIDE. |
| **Proprioceptive feedback** | During action execution, receptors in muscles, tendons, and joints continuously report position and force back to the brain. This closed loop enables smooth, accurate movement. | No feedback during action. `_act()` returns immediately with a constant. There is no continuous monitoring of action execution. | No sensorimotor loop. Actions are open-loop (fire and forget). |
| **Motor planning (SMA/PMC)** | Before movement begins, supplementary motor area and premotor cortex plan the sequence. Stage-dependent cerebrocerebellar communication occurs: cortex leads cerebellum during planning, cerebellum leads cortex during execution. | Planning happens in `_decide()` via world model rollouts, but the plan is never executed — it produces a single action index that goes to the stub. | Planning exists but execution does not carry out the plan. Like having a flight plan but no airplane. |
| **Basal ganglia action selection** | Selects which action to execute from competing alternatives through inhibition/disinhibition. Go/no-go pathways. | DecisionRouter does this well (reflex/habit/prefrontal cascade with endocrine modulation). But the selected action has no downstream execution path. | Selection works. Execution is the gap. |

---

## External State of Art Comparison

| Framework/Paper | Approach | Mae Comparison |
|---|---|---|
| **DreamerV3** (Hafner et al. 2023) | World model generates imagined trajectories, policy is trained in imagination, then actions execute in real environment with real rewards. The imagination-reality loop is closed. | Mae has a world model that can generate trajectories. But there is no "real environment" for actions to execute in. The loop is open. |
| **Subsumption Architecture** (Brooks, 1986 / 2025 revival) | Layered behavior-based control through sensorimotor loops. Each layer is a complete sense-act loop. Higher layers subsume lower ones. | Mae has the layer structure (reflex/habit/prefrontal in DecisionRouter). But the "act" end of each loop is a stub. The loops are sense-decide loops, not sense-act loops. |
| **Joint MLLM-WM Architecture** (2025) | Multimodal LLM handles semantic reasoning for task decomposition. World Model provides physics-based simulation. Actions are structured trajectories with causal dependencies between sub-goals. | Mae separates cognition (world model, causal engine) from execution. But has no structured action output — actions are single integers or dicts with no causal structure. |
| **Multi-Agent Embodied AI** (arXiv 2505.05108, 2025) | Real-world embodied AI must navigate complex scenarios where agents collaborate. Action execution involves physical effectors with continuous state feedback. Collaborative actions require synchronized execution. | Mae has collaboration via GNN communication and stigmergy, but these happen outside `_act()`. Agents cannot take coordinated actions — they can only coordinate their observations and decisions. |
| **SayCan** (Google, 2022) | LLM proposes multi-step sub-tasks, robot affordance function evaluates what is physically possible, and the robot executes the feasible sub-task. Action feasibility is checked against embodiment. | Mae has no affordance check. The world model proposes actions but there is no feasibility gate before (non-)execution. |

---

## Ranked Upgrade Recommendations

### Rank 1: Implement an Action Environment (CRITICAL)

**What it means:** Build the "room" where agents actually do things.

There needs to be an environment that changes when agents act. Currently, stigmergy is the closest thing Mae has, but markers are deposited as side-effects of learning and communication, not as direct consequences of action.

The environment could be:
- A task pool that agents claim, work on, and complete (returning real rewards)
- A grid/continuous space where agents move and interact
- A resource landscape where actions gather, transform, or distribute resources
- An abstract problem space where actions modify shared state

Without this, nothing else in the ACT step can work. This is the neuromuscular junction — the place where signal becomes force.

**Biological analogy:** Building the body. Right now Mae has a brain (cognition), a nervous system (communication), an immune system (defense), a memory (episodic memory), but no musculoskeletal system. She can think but cannot move.

### Rank 2: Override `_act()` in MycelialAgent with Triadic Execution

**What it means:** Replace the two-line stub with a three-phase execution cycle.

Biological motor execution has three phases:
1. **Motor planning** (translate decision to execution plan) — efference copy sent to cerebellum
2. **Execution** (change the environment) — signal reaches effector, force is applied
3. **Proprioceptive verification** (compare expected vs actual outcome) — reward signal generated from environment feedback

```
_act(action):
  1. PLAN: Generate execution plan from action + world model prediction (efference copy)
  2. EXECUTE: Apply action to environment, get actual outcome
  3. VERIFY: Compare prediction vs actual outcome -> generate reward + error signal
```

This makes ACT triadic (plan/execute/verify), gives it cause-effect power (action changes environment), and provides real reward signal for learning.

### Rank 3: Efference Copy Mechanism

**What it means:** When an agent acts, it simultaneously predicts what will happen.

Before executing, the world model generates a prediction of the next state and expected reward. After execution, the actual state and reward are compared to the prediction. The mismatch becomes:
- A cerebellar learning signal (world model training error)
- A surprise signal (for curiosity drive)
- A trust calibration signal (for validated imagination)

This is the single most biologically important mechanism missing from Mae. It is what makes action execution intelligent rather than mechanical.

### Rank 4: Fractal ACT at Every Scale

**What it means:** The ACT protocol should exist identically at every holonic level.

Currently:
- **Cell level:** `BaseAgent._act()` — stub
- **Tissue level:** `OctopusAgent.submit_task()` — submits to distributed cognition (different API)
- **Organ level:** `MorphogenesisCoordinator.handle_novel_problem()` — spawns agents (different API)
- **Organism level:** `MycelialModel.step()` — runs all agents (completely different)

Each of these should implement the same triadic ACT protocol (plan/execute/verify) through `holon_act()`, just at different scales with different effectors. The HolonMixin already defines `holon_act()` — it just needs real implementations at each scale.

### Rank 5: Action-Reward Coupling

**What it means:** Different actions should produce different rewards.

Currently, `_act()` returns 0.0 for every action. The only non-zero reward comes from VDN's global reward distribution, which distributes a sum of all agents' `last_reward` (which is always 0.0). This means all reward in Mae is currently zero.

The environment (from Rank 1) should return rewards that depend on:
- What action was taken
- What state the agent was in
- What other agents are doing (multi-agent effects)

This restores IIT's information postulate: different actions must have different consequences.

### Rank 6: Action Validation via Triad Enforcer

**What it means:** Actions should be validated before execution.

The TriadEnforcer validates processes through majority vote. Action execution is arguably the most critical process (it is where Mae affects the world). Yet it has zero validators registered.

Before an agent executes an action, the TriadEnforcer should validate it:
- Structural validator: Is the action compatible with the agent's current role?
- Behavioral validator: Is the action consistent with HAVEN risk assessment?
- Operational validator: Does the agent have the resources/capability to execute?

### Rank 7: Coordinated Multi-Agent Actions

**What it means:** Agents should be able to take joint actions.

Currently, agents act independently (Mesa's `shuffle_do`). There is no mechanism for two or more agents to execute a coordinated action — even though the communication infrastructure (GNN, stigmergy, quorum) is rich.

Coordinated actions would use the existing consensus mechanisms (collective dream, quorum sensing) to synchronize execution across multiple agents in the same step.

---

## Sources

- [Physiology, Motor Cortical - StatPearls (NCBI)](https://www.ncbi.nlm.nih.gov/books/NBK542188/)
- [Cerebellar circuit computations for predictive motor control (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12643008/)
- [Stage-dependent cerebrocerebellar communication during sensorimotor processing (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12494962/)
- [Efference copy in kinesthetic perception (J. Neurophysiology)](https://journals.physiology.org/doi/full/10.1152/jn.00545.2020)
- [Engaging distributed cortical and cerebellar networks through motor execution (Frontiers, 2023)](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2023.1165307/full)
- [Multi-agent Embodied AI: Advances and Future Directions (arXiv 2025)](https://arxiv.org/html/2505.05108v1)
- [A review of embodied intelligence systems: three-layer framework (Frontiers, 2025)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1668910/full)
- [Embodied AI Agents: Modeling the World (arXiv 2025)](https://arxiv.org/pdf/2506.22355)
- [Embodied AI: From LLMs to World Models (arXiv 2025)](https://arxiv.org/html/2509.20021v1)
- [IIT 4.0: Formulating the properties of phenomenal existence (PLOS Comp Bio, 2023)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011465)
- [How to be an integrated information theorist without losing your body (Frontiers, 2024)](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1510066/full)
- [IIT Wikipedia](https://en.wikipedia.org/wiki/Integrated_information_theory)