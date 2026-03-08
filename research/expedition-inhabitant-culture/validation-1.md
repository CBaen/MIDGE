# Validation Report: Expedition — Inhabitant Culture
## Validator: Independent Review
## Date: 2026-03-08

---

## Validation Methodology

All four findings files were read in full. The following codebase files were read directly:

- `mae_core/coordination/homeostasis.py` — verified setpoint count and API
- `mae_core/learning/curiosity.py` — verified `set_exploration_bonus()` and `get_exploration_targets()`
- `mae_core/market/resource_governor.py` — verified ResourceGovernor structure
- `mae_core/backbone/event_bus.py` — verified thread safety implementation

The following searches were performed to verify specific claims:

- Consumers of `coordination.homeostasis_correction` channel
- Callers of `set_exploration_bonus()` and `get_exploration_targets()`
- Existence and location of `StigmergicEnvironment`, `QuorumSpace`, `Stigmergy` classes
- Bootstrap assignment of `ctx.stigmergy` and `ctx.quorum_space`
- `StigmergicEnvironment` evaporation behavior
- Agent gradient-reading from stigmergy
- Mesa version installed (confirmed 3.4.2)
- `schedule_recurring` API usage in MIDGE (none found)
- `ResourceGovernor` registration in bootstrap
- `GovernanceLogger` existence (none found)
- `OrganBuilder` / `SenescenceManager` lifecycle coupling
- `GlobalWorkspace` existence and location
- `SomaticMap.get_critical_path()` existence
- `_decide()` cascade structure

---

## 1. Evidence Challenges

### Team 1 — Intrinsic Drives

**Challenge 1.1 — homeostasis_correction channel is NOT uncoupled**

Team 1 states: "Correction signals are emitted but nothing consumes them to change agent behavior." This is only partially accurate. `organism_state.py` subscribes to `coordination.homeostasis_correction` (line 191) and implements `_on_homeostasis_correction()` (line 276), which updates `self._homeostasis_deviation` tracking the maximum urgency seen. `connection_registrations.py` also registers the channel (line 338). The claim that "nothing consumes them" is wrong — `OrganismState` does consume corrections and tracks deviation.

However, the deeper claim remains valid: `_homeostasis_deviation` on `OrganismState` is tracked but does not feed into `_decide()`. The cascade code in `lifecycle_decision.py` calls `organism.get_reflex_override()` but there is no path from `_homeostasis_deviation` to a reflex override or advisory modulation. The orphaned-signal diagnosis is directionally correct but imprecise in saying "nothing subscribes."

**Challenge 1.2 — set_exploration_bonus() IS called autonomously from EndocrineSystem**

Team 1 states `set_exploration_bonus()` is "called nowhere autonomously." Code search shows it is called in:

- `bio_market_wiring.py` lines 222, 225, 231 — called from market event callbacks
- `endocrine_system.py` lines 429-430 — called when dopamine levels rise

This means CuriosityDrive IS being modulated externally through the EndocrineSystem's dopamine pathway. Team 1's framing that this method sits unused is incorrect. The genuine gap is narrower: the exploration bonus is modulated, but this modulation does not feed back into MIDGE-specific market investigation selection.

**Challenge 1.3 — The "no slot in _decide() for drive" claim needs precision**

Team 1 describes the `_decide()` cascade as "reflex → collision → stigmergy → advisory → worldline → dream." Code verification shows the actual cascade in `lifecycle_decision.py` is: reflex override from OrganismState → collision avoidance → stigmergy danger gradient → advisory+router → WorldlinePlanner → CollectiveDreamPlanner. The Team 1 description is accurate in structure. The cascade does have no homeostatic urgency slot, confirming that diagnosis. But the characterization that drives are "nowhere in the cascade" is not quite right — the reflex override from OrganismState IS the gateway where homeostatic urgency could enter (it already exists as a function call), it just doesn't yet act on homeostatic deviation.

**Challenge 1.4 — arXiv:2508.05619 citation mismatch**

Team 1 cites "arxiv:2508.05619 (2025) examines active inference in the era of LLM experience" as evidence for Active Inference/FEP momentum. However, the paper ID 2508.05619 was not independently verified. A paper dated August 2025 with this ID would be post-knowledge-cutoff and cannot be verified from the codebase. The AAMAS 2025 citation for factorised active inference is also cited without a verifiable conference proceedings page from internal evidence. This does not invalidate the FEP claim's validity, but the momentum evidence relies on sources that cannot be confirmed internally.

**Challenge 1.5 — "30-50 lines" estimate is speculative**

Team 1's Layer A implementation estimate of "30-50 lines across two files" has no basis beyond assertion. The actual scope depends on whether `HomeostasisRegulator.compute_correction()` returns a multi-dimensional vector (it currently returns individual per-parameter correction dicts), how `_decide()` receives the urgency (would require new state field on agent), and whether existing tests cover the touched files. This should not be presented as a confident estimate without code analysis.

---

### Team 2 — Emergent Culture

**Challenge 2.1 — Stigmergy and QuorumSpace DO exist in the codebase**

Team 2's most prominent claim is: "Stigmergy and QuorumSpace objects may not exist in `mae_core/emergent/`." This is a significant factual error. Verification shows:

- `StigmergicEnvironment` class exists at `mae_core/communication/stigmergy.py` line 42
- `QuorumSpace` class exists at `mae_core/communication/quorum_space.py` line 37
- `ctx.stigmergy = StigmergicEnvironment()` is set at `mae_core/bootstrap/foundation.py` line 131
- `ctx.quorum_space = QuorumSpace()` is set at `mae_core/bootstrap/foundation.py` line 132

Team 2 searched `mae_core/emergent/` when the classes live in `mae_core/communication/`. This is a directory search error that produced a false-negative. The implication — "the wiring is built, the object being wired to is absent" — is incorrect. Both objects are instantiated and available on `ctx`.

**Challenge 2.2 — Evaporation IS implemented in StigmergicEnvironment**

Team 2 states "Add evaporation step to Stigmergy (runs every 50 Mesa steps — fast enough to reflect signal decay)" as if evaporation does not exist. Code read of `stigmergy.py` shows `_decay_rate = 0.05` (default), `_apply_decay()` method at line 179, and decay applied via `math.exp(-self._decay_rate * marker.age)`. Evaporation is already built. The gap Team 2 identifies (calling evaporation periodically from a step hook) may be valid, but framing it as "add evaporation" misrepresents the current state.

**Challenge 2.3 — Agent gradient-reading gap is confirmed but partially wrong**

Team 2 states "no agent reads the trails." Code search finds `StigmergyMixin` in `mae_core/agents/mixins/stigmergy.py` IS part of `MycelialAgent`'s mixin stack (confirmed in `mycelial_agent.py` line 35). The mixin calls `self.stigmergy_env.get_gradient()` at line 107. So agents DO have the capability to read stigmergy gradients. The genuine gap may be that the gradient result doesn't influence final action selection (analogous to the homeostasis signal dissipation problem), not that reading is entirely absent.

**Challenge 2.4 — CollectiveDreamPlanner IS referenced in bootstrap**

Team 2 states CollectiveDreamPlanner "is referenced in bootstrap wiring" with uncertainty. Verification confirms it exists at `mae_core/cognition/collective_dream.py` and is imported in `bio_market_wiring.py` line 13. The collective_dream channel is registered in connection_registrations.py and used in the `_decide()` cascade (lifecycle_decision.py line 168-169). Team 2's hedged claim is confirmed but it is not as absent or uncertain as framed.

**Challenge 2.5 — "300x overhead penalty for LLM swarm agents" does not apply to MIDGE**

Team 2's synthesis correctly notes the 300x penalty doesn't apply to MIDGE's Python/rule-based agents. However, this framing is used to argue MIDGE can use classical stigmergy where LLM-based systems cannot. The distinction is valid, but it is not evidence that MIDGE's stigmergic coordination will work effectively — it is evidence that the cost is manageable. Effectiveness must be separately established.

---

### Team 3 — Self-Governance

**Challenge 3.1 — ResourceGovernor is NOT wired into bootstrap**

Team 3 states ResourceGovernor is "Built, wired to market APIs" in the mapping table. Code search finds ResourceGovernor defined only in `mae_core/market/resource_governor.py`. Searches across all bootstrap files find zero references to `ResourceGovernor`. Tests exist in `tests/test_resource_governor.py` but the governor is not instantiated in any bootstrap layer. Team 3's claim that it is "wired to market APIs" cannot be verified from the codebase — if it is wired, the wiring is not in the bootstrap modules. This is the most consequential factual error in Team 3's findings because the entire self-governance synthesis depends on ResourceGovernor being "already there."

**Challenge 3.2 — OrganBuilder does NOT listen to CH_SENESCENT**

Team 3 claims: "OrganBuilder listens and dissolves [the organ when CH_SENESCENT fires]." Search of `mae_core/morphogenesis/organ_builder.py` finds no references to `CH_SENESCENT` or `senescent`. The SenescenceManager publishes `CH_SENESCENT` (confirmed at senescence.py line 166), but nothing in the morphogenesis package subscribes to it. The lifecycle loop Team 3 describes (senescence → OrganBuilder dissolves → morphogenesis spawns replacement) is not wired. Presenting it as existing behavior that just needs "coupling" is inaccurate.

**Challenge 3.3 — Mesa 3.5 "schedule_recurring is production-ready" overstates certainty**

Team 3 states Mesa 3.5's `schedule_recurring` is "production-ready" with "no breaking changes from 3.4." Team 4 (who studied this more carefully) notes the interaction between MIDGE's custom `add_step_hook` API and Mesa 3.5's event scheduler is untested. Team 3 asserts production readiness without acknowledging this risk. The claim that `model.run_for(1)` is "functionally equivalent to `model.step()`" (cited from Team 4's findings) is accurate per Mesa documentation, but MIDGE's `MycelialModel` overrides `step()` with hook execution that may not be preserved in Mesa 3.5's internally-reimplemented step mechanism. This needs a prototype before asserting equivalence.

**Challenge 3.4 — GovernanceLogger described as "the one new file needed"**

Team 3 concludes the GovernanceLogger is the only genuinely new system needed. But given that ResourceGovernor is not bootstrapped, OrganBuilder → Senescence is not wired, and endocrine → resource coupling is absent, the scope of "wiring" is considerably larger than Team 3 implies. "Wiring existing systems" is doing significant load-bearing work in this synthesis. Three EventBus subscriptions will not complete a self-governance layer if the systems being subscribed to are not yet in the execution path.

**Challenge 3.5 — "Immune privilege can only be set during bootstrap" claim is architectural policy, not enforcement**

Team 3 states: "Criticality can only be set during bootstrap (the 33-layer bootstrap already defines a sealed order), not at runtime." This is presented as a safeguard against systems falsely marking themselves CRITICAL. However, `SomaticMap.update_criticality()` or equivalent runtime methods would need to be explicitly blocked. The bootstrap seal prevents new connections, but criticality level on a system already registered in SomaticMap could potentially be changed at runtime depending on the method visibility in `somatic_map.py`. Team 3 asserts this policy exists as enforcement, but it appears to be an architectural intent rather than a code-enforced constraint.

---

### Team 4 — Autonomous Scheduling & Scale

**Challenge 4.1 — "add_step_hook is MIDGE's own invention, not a Mesa 3.4 API" is confirmed and well-reasoned**

This is one of the most important and correctly verified findings across all four teams. Code confirms `add_step_hook` at `mae_core/model.py` line 253. This finding is accurate and has important implications for Mesa 3.5 migration: MIDGE's hooks are called BEFORE agent activation (line 119), and Mesa 3.5's event scheduler fires internally within `model.step()`. Whether MIDGE's hooks would run before or after Mesa 3.5's internally scheduled recurring events is genuinely unknown and needs testing.

**Challenge 4.2 — "model.run_for(1) is functionally equivalent to model.step()" cited without verification of MIDGE-specific behavior**

Team 4 cites this equivalence from Mesa documentation. However, in MIDGE's `MycelialModel`, `step()` is overridden to (1) call all `_step_hooks`, then (2) activate agents via ThreadPoolExecutor or shuffle_do. Mesa 3.5's internal reimplementation of step as an EventGenerator may not preserve this execution order if `MycelialModel.step()` is no longer the primary activation mechanism. This is a compatibility risk that Team 4 acknowledges but then proceeds to present the equivalence as established fact in the migration path.

**Challenge 4.3 — APScheduler 4.0 warning is appropriate but 3.11.2 version claim needs checking**

Team 4 warns "APScheduler 4.0 is pre-release and explicitly not production-ready. Use 3.11.2." This appears accurate based on public knowledge of APScheduler's release history. However, no PyPI version was confirmed from within the session. This is minor — the recommendation to use the stable 3.x branch is sound regardless of the exact version number.

**Challenge 4.4 — Thread count budget estimate is absent**

Team 4 states "50 daemon threads is well within range" and "200+ would start to show scheduling jitter" without any baseline measurement of how many threads MIDGE already creates. The OctopusColony uses daemon threads, Finnhub WebSocket uses a thread, the `ThreadPoolExecutor` maintains a pool. The total existing thread count is not reported, making the "50 more is fine" assertion incomplete.

---

## 2. Contradictions Between Teams

**Contradiction A: Who is responsible for coordinating market investigation tasks?**

- Team 2 proposes the QuorumSpace as the collective confidence oracle, replacing the convergence alerter's confidence formula.
- Team 3 proposes TriadEnforcer quorum votes for task claiming (before investigation begins).
- Team 4's InhabitantScheduler determines when each inhabitant fires.

These three systems operate on different phases of the investigation pipeline (pre-investigation scheduling → investigation claiming → post-investigation confidence). They are architecturally compatible but none of the three teams acknowledges the others' proposals in this area. The orchestrator needs to understand these as three distinct layers, not competing alternatives.

**Contradiction B: Mesa 3.5 as the scheduling solution**

Both Team 3 and Team 4 recommend Mesa 3.5's `schedule_recurring`. However:

- Team 3 presents it as "production-ready" and recommends upgrading Mesa as a key prerequisite.
- Team 4 explicitly warns the interaction between MIDGE's custom `add_step_hook` and Mesa 3.5's scheduler is "untested" and "needs verification."
- Team 4 also states `schedule_recurring` does NOT solve the wall-clock independence problem — it still requires daemon threads for true wall-clock inhabitants.

Team 3 adopts Team 4's finding (Mesa 3.5 headline feature) without adopting Team 4's caveats (compatibility risks, wall-clock limitation). This is selective citation that produces a misleadingly confident recommendation.

**Contradiction C: How much new code is required?**

- Team 1 synthesis: "The work is the connections, not new systems." Characterizes the gap as routing problems, not new construction.
- Team 3 synthesis: "Self-governance is not a new system — it is the activation of existing systems." Also connectivity-minimal.
- Team 3 then proposes GovernanceLogger as a new file, and its synthesis identifies 4 layers of work.
- Team 2 proposes implementing Stigmergy + QuorumSpace as "proper objects" — but they already exist. This reveals that Team 2's gap analysis was based on a flawed code search (looking in `emergent/` instead of `communication/`).

The general convergence on "this is mostly wiring" is probably correct in spirit but understated in scope. The actual missing wiring (endocrine→resource, senescence→OrganBuilder, homeostasis urgency→_decide) represents non-trivial work with blast radius implications.

---

## 3. Alignment Drift

**Drift 3.1 — Teams underweight the "50+ autonomous inhabitants" scope**

The Research Brief's Expected Outcome reads: "Systems wake up on their own schedules. Some are curious and chase partial signals. Some are cautious and patrol for threats. They coordinate through pheromone trails and quorum sensing." This implies meaningfully differentiated inhabitant personalities.

Team 2 proposes addressing this through history-driven differentiation (novel approach 3). All teams acknowledge the stem cell principle (Law 5). But none of the teams does the numerical work: MIDGE currently has 12 Mesa agents + 3 OctopusArms = 15 entities with any agency. Getting to 50+ requires either spawning new Mesa agents (morphogenesis path) or wiring existing biological systems to behave with more autonomy without Mesa agent instantiation. Team 4's InhabitantScheduler could enable biological systems to fire on their own clocks, which would satisfy "waking up on own schedules" without creating 50 new Mesa agents. But Team 4 never explicitly addresses how a biological system (not a Mesa agent) becomes an "inhabitant" with its own schedule. This is the core architectural gap the Research Brief is asking about, and all four teams circle it without landing on it.

**Drift 3.2 — Team 1's Layer C (LearningProgressMonitor) drifts toward RSI Layer 4 territory**

Team 1 proposes a `LearningProgressMonitor` that injects investigation goals into a `TaskPool` based on PatternLibrary confidence intervals. This is architecturally in the PatternArchaeology / RSI Layer 4 territory, not strictly in the "intrinsic drive architecture" domain. It is a valuable proposal, but it extends into a territory the Research Brief designates as "do NOT redesign pattern archaeology." The distinction between injecting goals INTO the archaeology system versus designing new archaeology behavior needs explicit flagging.

**Drift 3.3 — Team 3's HPA cascade mapping uses GlobalWorkspace as "Hypothalamus"**

Team 3 maps: GlobalWorkspace (Hypothalamus) → EndocrineSystem (Pituitary) → ResourceGovernor (Adrenal). The GlobalWorkspace in MIDGE's codebase (`mae_core/patterns/global_workspace.py`) is specifically the GWT attention broadcast system for pattern recognition — it is not a general-purpose coordination hub. Using it as the HPA-axis "Hypothalamus" to drive resource governance creates an architectural dependency between two systems that currently have separate responsibilities. This is not prohibited, but it is a consequential architectural coupling that Team 3 does not flag as a design decision requiring Guiding Light input.

**Drift 3.4 — Team 4 proposes "InhabitantScheduler as Layer 34" but the Research Brief prohibits breaking the 33-layer bootstrap order**

Team 4 suggests: "Wire it into bootstrap Layer 33 (or Layer 34 if the sequence matters)." The Research Brief's destructive boundaries state "Do NOT break the 33-layer bootstrap order." Layer 34 is a de facto new layer. Team 4 acknowledges "the sequence matters" but does not resolve how this constraint is satisfied. This requires explicit resolution before implementation.

---

## 4. Missing Angles

**Missing 4.1 — No team examined whether OrganBuilder actually creates new inhabitants**

The morphogenesis package exists (`mae_core/morphogenesis/organ_builder.py`, `coordinator.py`, `reproductive_system.py`). Team 3 references OrganBuilder for inhabitant lifecycle but does not verify whether it can spawn entirely new inhabitant types or only replicate existing organ configurations. The distinction is critical: can MIDGE grow a genuinely new inhabitant with a different signal focus, or can it only replace existing ones? This question is central to the "50+ inhabitants from within" vision and no team investigated it.

**Missing 4.2 — No team verified ResourceGovernor bootstrap status before building on it**

Three of four teams treat ResourceGovernor as an active governor in the running system (Team 3 extensively, Teams 1 and 2 tangentially). None searched for its bootstrap instantiation. Verification reveals it is defined only in `mae_core/market/resource_governor.py` and referenced only in its test file — it is not bootstrapped in any layer. If ResourceGovernor is not in the execution path, the entire "three EventBus subscriptions complete self-governance" proposal has a missing foundation.

**Missing 4.3 — No team explored what "inhabitant" means architecturally in MIDGE**

The Research Brief asks: "How do we build a culture of inhabitants?" None of the teams defined what the unit of inhabitation is. Is it:
(a) A Mesa `MycelialAgent` instance (existing pattern)?
(b) A biological system (HomeostasisRegulator, CuriosityDrive) that gets scheduled by InhabitantScheduler?
(c) An OctopusArm (existing P2P agent)?
(d) A new `Inhabitant` class wrapping any of the above?

Without this definition, the teams recommend mechanisms (drives, quorum, scheduling) without specifying what they will be attached to. Team 4 is closest to addressing this but only in the context of wall-clock scheduling, not inhabitant identity.

**Missing 4.4 — No cross-team simulation of what happens at step 1 of the new system**

No team described what MIDGE would look like at step 1 after implementing their recommended approach. What fires? What gets published? What does an agent do differently? For a system that is largely "wiring existing components," the absence of a concrete first-step scenario makes it impossible to validate that the architecture actually produces autonomous behavior rather than more elaborate reactivity.

**Missing 4.5 — Law 1 compliance for new connections not fully analyzed**

Law 1 requires every connection A↔B to have a witness C. Team 3's "three EventBus subscriptions" (GlobalWorkspace→EndocrineSystem→ResourceGovernor) is a triadic chain, but each individual subscription is a dyad. Adding GlobalWorkspace→ResourceGovernor as a direct subscription (to complete a K3 triad) is not mentioned. Team 1's "add a consumer on coordination.homeostasis_correction that writes urgency to each agent" creates a new HomeostasisRegulator→Agent dyad without identifying a triadic witness. All teams should have applied Law 1 to their proposed new connections.

---

## 5. Agreements (High-Confidence Zones)

The following findings appeared independently across multiple teams and are therefore the most trustworthy:

**Agreement A — The fundamental gap is drive-to-action coupling, not drive generation**

Teams 1, 2, and 3 all independently identify that MIDGE has the biological organs but not the behavioral outputs. Team 1: "drive routing is missing." Team 2: "EndocrineSystem broadcasts the cultural mood... agents are not reading it." Team 3: "the cross-system coupling is not wired." This consensus is the strongest finding of the expedition. The organism has senses and effectors; the nervous connections between them are incomplete.

**Agreement B — Mesa 3.5 schedule_recurring is the right in-step cadence solution**

Teams 3 and 4 independently recommend Mesa 3.5's `schedule_recurring` for replacing `step % N` branches. Team 4 provides the critical caveat (untested with MIDGE's hook system). The agreement on the API is solid; the readiness for implementation is not.

**Agreement C — OctopusColony's monitoring thread is the prototype for wall-clock inhabitants**

Teams 2 and 4 independently identify the OctopusColony's daemon monitoring thread as the working example of wall-clock-independent execution. Team 4 explicitly calls it "the prototype for everything that follows." This is the right reference point.

**Agreement D — Endocrine system is the cultural medium for collective mood**

Teams 1, 2, and 3 all identify the EndocrineSystem as the chemical broadcast medium that agents should read. The hormones are already defined, the channel exists, agents don't yet consume them to modify behavior. This specific wiring (EndocrineSystem → agent behavior modulation) appears in all three teams' findings and is the highest-priority single connection.

---

## 6. Surprises

**Surprise 1 — Stigmergy and QuorumSpace are further along than any team knew**

The most striking finding from code verification: the Stigmergy and QuorumSpace objects are not absent or stubbed — they are fully instantiated at bootstrap (`foundation.py` lines 131-132), have working deposit and gradient methods, and `StigmergicEnvironment` has working exponential decay built in. Furthermore, `StigmergyMixin` is part of `MycelialAgent`'s mixin stack, meaning agents CAN read stigmergy gradients today. Team 2's entire "Tier 1" recommendation (implement these as proper objects, add evaporation) is solving a problem that is already solved. This discovery would redirect implementation effort from building to activating — connecting the gradient read result to final action selection.

**Surprise 2 — ResourceGovernor has no bootstrap presence**

The inverse surprise: ResourceGovernor, which all teams treat as an active system, is not bootstrapped into any execution layer. It exists as a class, has tests, but is not instantiated in any layer of `mae_core/bootstrap/`. This is a significant gap none of the teams caught, and it undermines Team 3's synthesis substantially. The first step toward resource self-governance is bootstrapping the ResourceGovernor, not wiring it to the EndocrineSystem.

**Surprise 3 — The _decide() cascade has a CollectiveDreamPlanner slot**

Team 1 describes the `_decide()` cascade as ending at "dream" as the last resort. Code verification confirms the CollectiveDreamPlanner IS in the cascade at `lifecycle_decision.py` line 168. This suggests the cascade is already more sophisticated than Team 1's table implies — the "dream" step provides a consensus-based fallback. This is a positive surprise: the cascade already has an emergent collective intelligence slot at its base level.

**Surprise 4 — HomeostasisRegulator correction signals DO have a consumer**

Team 1's diagnosis that "nothing subscribes" to homeostasis corrections is incorrect. `OrganismState` subscribes and tracks `_homeostasis_deviation`. The real gap is narrower: this deviation value is tracked but not yet exposed to `get_reflex_override()` which is the existing gateway into `_decide()`. This is a smaller surgical connection than Team 1's analysis implies — modify `get_reflex_override()` to check `_homeostasis_deviation` and return an action when urgency exceeds threshold. Team 1's proposed solution (add a new consumer) may be less correct than simply activating the existing consumer.

---

## Summary Assessment by Team

| Team | Factual Accuracy | Alignment | Completeness | High-Confidence Findings |
|------|-----------------|-----------|--------------|--------------------------|
| Team 1 (Intrinsic Drives) | Good — errors on set_exploration_bonus and homeostasis consumer; diagnosis directionally correct | Good | Gaps section is honest | Drive-to-action coupling gap; Layer A/B/C architecture |
| Team 2 (Emergent Culture) | Poor — major directory search error led to false-negative on Stigmergy/QuorumSpace existence | Good | Tier structure useful | Quorum as confidence oracle; pheromone multi-layer decay rates |
| Team 3 (Self-Governance) | Moderate — ResourceGovernor bootstrap gap missed; OrganBuilder→Senescence coupling assumed not verified | Good — HPA mapping useful | Gaps section strongest across all teams | HPA mapping concept; GovernanceLogger as needed new file |
| Team 4 (Scheduling/Scale) | Strong — most carefully code-verified team | Good | Warning about wall-clock/step boundary is critical | Two-tier architecture; OctopusColony as prototype; Mesa 3.5 caveats |

---

## Recommended Priority Order for the Orchestrator

Based on what the code actually shows, and ranked by confidence in the evidence:

1. **Bootstrap ResourceGovernor** — it is built and tested but not in the execution path. This is prerequisite to all governance work.

2. **Activate OrganismState homeostasis urgency → `get_reflex_override()`** — the consumer already exists (OrganismState), the gateway already exists (reflex_override in _decide()). This is a surgical 5-10 line change, not a new system.

3. **Wire endocrine → behavior modulation** — agreed by Teams 1, 2, 3 as highest-leverage single connection. EndocrineSystem → agent `exploration_bonus` via `set_exploration_bonus()` is already proven callable.

4. **Add quorum threshold check to ConvergenceAlerter** — QuorumSpace exists, deposits happen, the threshold logic does not. This is the activation Teams 1-3 converge on.

5. **Verify Mesa 3.5 compatibility with MIDGE's step hook system via prototype** — before any migration.

6. **Build InhabitantScheduler** (Team 4's proposal) — after verifying Mesa 3.5 interaction, since it may be subsumed by Mesa 3.5's scheduler.

7. **Build GovernanceLogger** — Team 3's one genuinely new file. Prerequisite: ResourceGovernor must be bootstrapped first.

---

*Validation performed 2026-03-08. All code references verified by direct file reads and grep searches.*
