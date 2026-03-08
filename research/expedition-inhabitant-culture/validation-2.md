# Validation Report: Expedition — Inhabitant Culture
## Date: 2026-03-08
## Validator: Independent Review

---

## Protocol Note

This report follows divergence-first order: challenges, contradictions, and alignment failures before agreements. Every codebase claim below has been verified against the actual files.

---

## 1. Evidence Challenges

### 1.1 Team 2: Stigmergy and QuorumSpace claimed as possibly absent — WRONG

Team 2's most prominent finding — "the Stigmergy and QuorumSpace objects may not exist" — is incorrect. Both files exist and are fully initialized.

**Verified:**
- `StigmergicEnvironment` lives at `C:\Users\baenb\projects\MIDGE\mae_core\communication\stigmergy.py`
- `QuorumSpace` lives at `C:\Users\baenb\projects\MIDGE\mae_core\communication\quorum_space.py`
- Both are instantiated in `mae_core\bootstrap\foundation.py` lines 131-132: `ctx.stigmergy = StigmergicEnvironment()` and `ctx.quorum_space = QuorumSpace()`
- Both are registered with `SomaticMap` in `mae_core\bootstrap\wiring.py` line 471

Team 2 looked in `mae_core/emergent/` — the wrong directory. The objects live in `mae_core/communication/`. Team 2 acknowledged graceful None-handling as evidence they might not exist, but that defensiveness exists to handle import failures, not object absence. This is a critical factual error that undermines Team 2's Tier 1 recommendations (both are framed as "implement what doesn't exist").

**Impact:** Team 2's #1 recommendation — "Implement Stigmergy and QuorumSpace as proper objects" — is work that has already been done. The actual gap is not existence but evaporation step activation and agent read integration.

### 1.2 Team 3: "EndocrineSystem — hormones not consumed" is partially wrong

Teams 2 and 3 both claim hormones are published but nothing consumes them behaviorally. This is overstated.

**Verified from `mae_core\bootstrap\wiring.py` line 412-413:**
```
ctx.endocrine.register_threat_detector(ctx.threat_detector)
ctx.endocrine.register_auto_healer(ctx.auto_healer)
```

**And from `wiring.py` lines 440-457:** A `_on_hormone_state_update` callback is registered on `endocrine.state_update` that reads dopamine and melatonin and modulates `SignalPriorityResolver._urgency_map` and `_config.budget_per_step`.

So three behavioral consumers do exist: ThreatDetector (cortisol sensitivity), AutoHealer (cortisol priority), and SignalPriorityResolver (dopamine/melatonin). The claim that "hormones published on EventBus but nothing changes agent behavior" is an overstatement. What is correctly missing: **ResourceGovernor is not a hormone consumer** (confirmed — no match for `ResourceGovernor` in `endocrine_system.py`), and individual agent `_decide()` behavior is not hormone-modulated.

Teams 2 and 3 should have acknowledged the partial wiring before declaring the system entirely decoupled.

### 1.3 Team 3: Mesa 3.5 `schedule_recurring` — version claim needs qualification

Team 3 and Team 4 both recommend upgrading to Mesa 3.5 and using `schedule_recurring`. Mesa 3.4.2 is confirmed installed. Mesa 3.5's `schedule_recurring` is referenced as production-ready and "headline feature."

**What is not verified:** Team 4 correctly notes the interaction between `add_step_hook` (MIDGE's own invention, confirmed at `model.py` line 253) and Mesa 3.5's event scheduler is untested. Team 3 presents Mesa 3.5 as straightforwardly drop-in. Team 4 is more honest here. Neither team ran the 4,536-test suite against Mesa 3.5 to confirm "no breaking changes." Both are citing release notes, not tested behavior in MIDGE's specific setup.

### 1.4 Team 1: IMGEP source citation mismatch

Team 1 cites IMGEP from two sources and conflates them: `arXiv:1708.02190` (the 2017 preprint) and `dl.acm.org/doi/10.5555/3586589.3586741` (the JMLR 2022 volume). The JMLR link points to JMLR Vol 23, 2022 — but the DOI `10.5555/3586589.3586741` does not resolve to a publicly verifiable URL without institutional access. The claim that it was "Demonstrated on real humanoid robot" applies to the 2022 version, but the open-source code linked (`github.com/sebastien-forestier/IMGEP`) is for the 2017 variant. These are related but distinct works. The research is valid; the citation accuracy is sloppy.

### 1.5 Team 1: D2A GitHub link — unverifiable as of research date

Team 1 cites `github.com/zfw1226/D2A` as an open-source implementation. This link was not independently verified. The arXiv paper (2412.06435) was published December 2024 and the claim of an open-source implementation is plausible but should be flagged as "cited, not confirmed."

### 1.6 Team 3: GaaS paper (arXiv:2508.18765) is 2025 — not peer-reviewed

Team 3 acknowledges this as a maturity risk but presents it as a "pattern validation" for MIDGE's existing architecture. This is valid reasoning (the pattern pre-dates the paper), but the citation itself adds nothing over directly noting that MIDGE's existing ResourceGovernor already implements non-invasive audit patterns. The paper is being used for credibility it hasn't earned through peer review.

---

## 2. Contradictions Between Teams

### 2.1 Endocrine coupling — Teams 1, 2, and 3 give contradictory framing

- **Team 1** frames the gap as: biological signals don't reach `_decide()` in agents (drive-to-action coupling missing).
- **Team 2** frames the gap as: EndocrineSystem broadcasts but nothing changes agent behavior.
- **Team 3** frames the gap as: endocrine → ResourceGovernor coupling is missing.

All three are correct about different gaps, but none acknowledges the others' framing. The actual situation has three distinct missing couplings: (1) hormone → ResourceGovernor throttling (Team 3's gap), (2) hormone → individual agent `_decide()` priority (Team 1's gap), (3) hormone → cultural norm imitation weight (Team 2's gap, re: oxytocin → peer trust). These are three separate wires, not one. Whoever synthesizes must treat them as independent, not as one "endocrine coupling" task.

### 2.2 Teams 3 and 4 give conflicting Mesa 3.5 advice

- **Team 3** recommends Mesa 3.5 `schedule_recurring` as the right mechanism for inhabitant heartbeats (Approach 8), calling it "production-ready" and the "right mechanism."
- **Team 4** recommends Mesa 3.5 only for Tier 1 (step-relative cadences) and explicitly rejects it as the wall-clock solution. Team 4 also flags the untested interaction between `add_step_hook` and Mesa 3.5's event scheduler.

Team 4's position is better-grounded. Team 3 appears to have adopted the Mesa 3.5 recommendation without verifying MIDGE's specific hook architecture. Team 4's explicit flag about the `add_step_hook` interaction is the more honest assessment — confirmed valid since `add_step_hook` exists in MIDGE's own `MycelialModel` (line 253 of `model.py`), not in Mesa's base class.

**The contradiction matters:** if Team 3's recommendation (use `schedule_recurring` for all heartbeats) is followed without understanding Team 4's caveat, the step hook interaction is a real risk.

### 2.3 Teams 2 and 4 on OctopusColony specialization

- **Team 2** reports: "Colony exists; specialization is GENERAL for all members — no role differentiation."
- **Team 4** uses the OctopusColony's `_monitoring_loop` as the prototype for `InhabitantScheduler`, implying it's a mature pattern.

Neither is wrong, but neither acknowledges the other's constraint. If OctopusColony arms are all GENERAL (Team 2's finding), the monitoring thread prototype Team 4 references is for health monitoring only, not for domain-differentiated inhabitants. Building InhabitantScheduler on top of still-undifferentiated OctopusArms is building the execution substrate while leaving the role problem unsolved.

---

## 3. Alignment Drift

### 3.1 Team 1: Drive modulation vs. autonomous initiation — brief asks for autonomous wakeup

The Research Brief's expected outcome: "Systems wake up on their own schedules." Team 1's synthesis recommends starting with drive **modulation** (changing existing agent behavior) before drive **initiation** (biological systems generating their own EventBus messages). Team 1 explicitly defends this as "safer."

This is a reasonable incremental path, but it is a partial answer to the Brief. Modulation produces more-reactive agents, not autonomous inhabitants. An agent that wakes up on the Mesa step loop and merely has a different priority score is not the same as an inhabitant that wakes on its own clock. Team 1 acknowledges this but frames it as a "start here" rather than flagging it as incomplete relative to the Brief's vision.

The Brief explicitly says: "They don't wake up on their own, pursue their own goals, or coordinate with each other autonomously" — this is the stated deficiency. Team 1's recommended Layers A+B (homeostatic urgency + drive-weighted advisory) address goal prioritization only, not autonomous wakeup. Layer C (learning progress goals) begins to address autonomous initiation but only for excavation targets, not for all system types.

### 3.2 Team 3: GovernanceLogger is a new system — needs approval per CLAUDE.md rules

Team 3 recommends creating `mae_core/governance/governance_logger.py` as the "one genuinely new system needed." The project's CLAUDE.md rules state: "Tool Creation: New scripts, skills, agents, or hooks require Guiding Light's explicit approval."

This is the only explicit new-file creation recommended across all four teams. Teams 1, 2, and 4 all frame their recommendations as wiring existing systems. Team 3 should have explicitly flagged this as requiring Guiding Light approval rather than burying it in a synthesis recommendation.

### 3.3 Team 4: InhabitantScheduler is also a new system

Team 4 recommends building `InhabitantScheduler` ("50 lines of code, modeled on OctopusColony's monitoring loop"). Same issue as Team 3's GovernanceLogger — this is a new file, not a wiring of existing code, and requires explicit approval. Team 4 frames it as "50 lines" to minimize it, but "50 lines" is still a new system. The CLAUDE.md rule applies regardless of size.

### 3.4 Team 2: "Deploy 50 identical stem cells" — misaligns with current population

The Brief says MIDGE currently has 12 Mesa agents plus 3 OctopusArms. Team 2's synthesis concludes "deploy them identical, seed them differently" but gives no path from 15 to 50+ agents. This scaling question is nominally Team 4's domain, but Team 4 also does not address how 50+ agents are actually spawned. The Brief explicitly frames this as the problem ("How do we build a culture of inhabitants?") and neither team gives a concrete mechanism for bootstrapping from 15 to 50+. This is a gap in the collective research output.

### 3.5 Team 3: "Populate HolonRegistry as org chart" — assigned to wrong layer

Team 3 says populating the three-division hierarchy "belongs in `mae_core/bootstrap/market.py` around Layer 33." The Brief's constraint says "Do NOT break the 33-layer bootstrap order." Adding new parent holon registrations at Layer 33 risks changing bootstrap behavior for the entire organism. This should be flagged as potentially order-sensitive, not presented as a simple data population task.

---

## 4. Missing Angles

### 4.1 No team addressed the drive-to-action gap in non-Mesa-agent bio systems

Team 1 correctly notes that CuriosityDrive and HomeostasisRegulator are not `MycelialAgent` subclasses — they don't have `step()` methods. But none of the four teams addressed the architectural question: should biological systems be given `step()` methods and registered as Mesa agents? Or should they remain non-agent systems that emit EventBus messages? This is the core fork in the architecture and nobody resolved it.

The question has real consequences for Teams 3 and 4: if bio systems become Mesa agents, the Mesa scheduling path (Teams 3 and 4) applies to them. If they remain non-agent systems, they must use daemon threads or EventBus emission to initiate behavior. None of the four teams named this choice explicitly.

### 4.2 No team asked whether `add_step_hook` is the right abstraction at scale

Team 4 counts 50+ registered step hooks as a coming bottleneck, but nobody questioned whether `add_step_hook` (MIDGE's own invention, confirmed in `model.py`) should be replaced, extended, or kept. At 50+ inhabitants each potentially registering hooks, the flat list of hooks-that-fire-every-step becomes an O(N) overhead per step regardless of cadence gating. The right architectural question is: should MIDGE's `MycelialModel` acquire its own priority-queue-based scheduler instead of the flat hook list? This would be a bigger change but solve the scaling problem more cleanly than Mesa 3.5 `schedule_recurring`.

### 4.3 No team verified whether sensing_hook.py `step % N` branching is in a step hook

All four teams discuss the `step % N` pattern as being in `_market_sense_hook` (a step hook). This was confirmed — `sensing_hook.py` has a `_step_counter` variable and 9+ `step_counter % N` branches. But the teams did not verify: how much of MIDGE's cadence logic is in hooks vs. in agent `step()` methods vs. in `market_hooks.py`. This matters because Mesa 3.5 `schedule_recurring` applies to agent activation, not to step hooks — migrating hooks is a different operation than migrating agent step() cadences.

### 4.4 No team addressed test coverage implications

The Brief requires zero regressions against 4,536 tests. Teams propose wiring HomeostasisRegulator → `_decide()`, EndocrineSystem → ResourceGovernor, Stigmergy evaporation step, and Mesa 3.5 upgrade. None of the four teams discussed which existing tests would need to change, which new tests would be required, or where conftest.py isolation (which already exists for Thompson/calibrator/outcome data per the MEMORY.md) would need to extend to new systems like the hormone modulation paths.

---

## 5. Agreements — High-Confidence Zone

The following findings appear independently across multiple teams, increasing confidence:

### 5.1 The gap is wiring, not capability (All 4 teams)

Every team converges on the same diagnostic: MIDGE has the biological organs but they are not connected to behavior. Team 1: "the problem is mostly activation, not absence." Team 2: "the culture emerges when these systems are connected." Team 3: "the gap in every row is the same: cross-system coupling not wired." Team 4: "the path to 50+ autonomous inhabitants is a migration, not a rewrite."

This is the highest-confidence finding of the expedition. The implication is important: the build phase should be primarily wiring work with targeted new-file additions, not a new architecture.

### 5.2 Hormones publish but ResourceGovernor doesn't subscribe (Teams 2 and 3)

Both independently identify that EndocrineSystem → ResourceGovernor coupling is absent. Confirmed: no `ResourceGovernor` reference appears in `endocrine_system.py`. The cortisol → budget tightening loop does not close.

### 5.3 add_step_hook is MIDGE's own invention, not Mesa's (Team 4, confirmed)

Verified at `model.py` line 253: `add_step_hook` is defined on `MycelialModel`, not inherited from Mesa. This matters for the Mesa 3.5 migration path — MIDGE cannot assume Mesa 3.5's internals interact cleanly with its own hook system.

### 5.4 The OctopusColony monitoring thread is the prototype for wall-clock inhabitants (Team 4)

Confirmed in `octopus_colony.py` lines 284-296 and 386-396: `threading.Thread(target=_monitoring_loop, daemon=True)` with `time.sleep(self._monitoring_interval)`. This exact pattern is already proven in production. Team 4's recommendation to generalize it is well-grounded.

### 5.5 SenescenceManager exists and is ready for lifecycle governance (Team 3)

Glob confirms `mae_core/emergent/senescence.py` exists. Team 3's description of it matches what would be expected from the MEMORY.md notes. The wear-accumulation mechanism is there; CircadianRhythm coupling is missing.

---

## 6. Surprises

### 6.1 Stigmergy and QuorumSpace are more complete than expected

The most surprising finding from code verification: `StigmergicEnvironment` and `QuorumSpace` not only exist but are full implementations with decay rates, thread safety, concentration tracking, and contributor attribution. They are also registered in SomaticMap. This means MIDGE's stigmergic coordination infrastructure is further along than any of the four teams realized. The actual gap is: (1) no evaporation step runs in the current step loop (nobody calls `StigmergicEnvironment.decay_all()` periodically), and (2) agents don't read gradient maps before selecting investigation targets. These are smaller gaps than the teams implied.

### 6.2 The endocrine system already has a partial behavioral connection

The discovery that `SignalPriorityResolver` is hormone-modulated via `wiring.py:_on_hormone_state_update` (dopamine and melatonin affecting budget_per_step) means hormones already reach one market behavior. This was missed by all four teams. It also means the "first connection" between hormones and behavior already exists as a tested pattern to extend from, not a blank canvas to start on.

### 6.3 sensing_hook.py's `_step_counter` is NOT step % N — it's a local counter

The `sensing_hook.py` uses `self._step_counter` incremented internally, not `model.steps` or a shared counter. This means Mesa 3.5 `schedule_recurring` cannot directly replace these cadence gates without a refactor — the hook is already a stateful object with its own counter. Team 4's proposed Mesa 3.5 migration would require changing the hook's internal design, not just replacing branches with schedule registrations.

---

## Summary Verdict

| Finding | Assessment |
|---------|-----------|
| Team 1 drive architecture research | Strong. HRRL, IMGEP, subsumption well-sourced. Three-layer synthesis (A+B+C) is the right shape. Modulation-before-initiation is defensible but undersells the Brief's autonomous wakeup requirement. |
| Team 2 emergent culture research | Good overall research; weakened by the false stigmergy/quorum absence claim, which erroneously positions implemented work as a top priority. The synthesis insight — "culture is activation of dormant systems" — survives the error. |
| Team 3 self-governance research | Best-structured team. HPA cascade mapping is the strongest single recommendation. GovernanceLogger and Mesa 3.5 heartbeat advice need qualification. Endocrine consumer claim is overstated. |
| Team 4 scheduling/scale research | Most technically rigorous. add_step_hook claim verified correct. Two-tier architecture (Mesa 3.5 + InhabitantScheduler) is sound but InhabitantScheduler requires approval. The `_step_counter`-vs-Mesa issue deserves more attention. |
| Collective gap | No team addressed the bio-system-as-agent-vs-non-agent architectural fork. No team gave a concrete path from 15 to 50+ inhabitants. |

**Highest-priority corrections before build planning:**
1. Remove Team 2's Tier 1 item 1 ("Implement Stigmergy and QuorumSpace") — already done. Replace with "Implement evaporation step for StigmergicEnvironment and wire gradient reads into OctopusArm task selection."
2. Flag GovernanceLogger (Team 3) and InhabitantScheduler (Team 4) as requiring Guiding Light approval.
3. Clarify whether bio systems (HomeostasisRegulator, CuriosityDrive) should become Mesa agents or remain EventBus emitters — the entire drive-to-action strategy depends on this fork.
4. Add test coverage plan to any wiring recommendation that touches `_decide()` cascade or ResourceGovernor.
