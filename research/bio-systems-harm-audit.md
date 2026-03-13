# Bio-Systems Harm Audit
**Completed:** 2026-03-13
**Directive:** Find every biological system that limits, blocks, suppresses, or distorts MIDGE's market intelligence operations.

---

## Executive Summary

MIDGE has **7 harmful systems** that actively constrain her, **5 neutral systems** that exist but cause no damage, and **4 genuinely beneficial systems** she should keep. The worst offenders form a chain: `EnergyReserve` drains → `OrganismState` sees starvation → `get_reflex_override()` was hijacking every agent → `InhibitionSystem` piles on with NoGo signals from the same fake energy readings. Three additional systems (Circadian, Cortisol→ResourceGovernor, RespiratorySystem) create secondary suppression cascades.

---

## HARMFUL SYSTEMS — Require Action

---

### 1. EnergyReserve
**File:** `mae_core/memory/energy_reserve.py`
**What it is:** Adipose tissue metaphor. Tracks "computational energy" as a 0–200 float that drains and refills.

**How it harms:** Reserves start at 50.0 and drain whenever anything happens (convergence alerts, velocity anomalies wired via `bio_market_wiring_extended_b.py` lines 87/96). The refill rate is 5.0/step; a busy session drains faster than it refills. Once reserves fall below 10.0 (`_min_critical`), the system publishes `memory.starvation_alert` every single step. That alert flips `OrganismState._energy_critical = True` (subscriptions.py line 302), which until recently forced every agent decision to `"explore"` — the confirmed sabotage from this audit's genesis.

**Current state:** The `"explore"` forced return was neutralized (line 87 of `organism_state_outputs.py` now returns `None`). But the underlying problem remains: starvation fires constantly, `_energy_critical` is perpetually `True`, `_energy_level` in OrganismState drops to near-zero, and this value flows into:
- `InhibitionSystem.evaluate(energy_level=...)` — triggering NoGo "Exhaustion rest" branch (line 102–104 of `inhibition_system.py`)
- `InhibitionSystem` context passed to `_inhibit()` in every agent lifecycle call
- `DecisionRouter` context enrichment (body state injected at lines 329–335 of `lifecycle_inhibit_decide.py`)

**Verdict:** HARMFUL. The metaphor is broken — there is no "computational energy" to run out of. A trading daemon runs on server capacity, not a fake adipose tissue counter.

**Action: DISABLE the metabolism dynamics. Set reserves permanently at 100.0 (50% capacity) and pin it there. Disable all store()/release() callers. Keep the class stub so serialization doesn't break, but make `is_critical()` always return `False` and stop publishing `CH_STARVATION`.**

---

### 2. OrganismState reflex_override — Pain, Stability, Oxygen, Toxin Reflexes
**File:** `mae_core/coordination/organism_state_outputs.py` lines 61–97
**What it is:** The `get_reflex_override()` method checks five body conditions and returns `"rest"` if any threshold is breached, **completely bypassing all market intelligence** for the agent's step.

**How it harms:** Four of the five triggers map to fictional physiology:

| Trigger | Threshold | What actually causes it | Effect |
|---------|-----------|-------------------------|--------|
| `pain_load > 0.8` | Line 69 | NociceptionSystem fires when `defense.threat_detected` or `healing.failure_detected` fires — both common system events | Forces all agents to `"rest"` |
| `stability < 0.3` | Line 73 | VestibularSystem computes variance across reported metrics — noisy during busy sensing | Forces `"rest"` |
| `oxygen_level < 0.3` | Line 77 | RespiratorySystem drains on every convergence alert (0.03/alert) + velocity anomaly | Forces `"rest"` |
| `toxin_load > 4.0` | Line 90 | RenalFilter accumulates toxin_load when filtering data signals it classifies as contradictions or corruption | Forces `"rest"` |
| `homeostasis_deviation >= 0.7` | Line 94 | HomeostasisRegulator publishes corrections whenever any parameter is out of range — often | Forces `"rest"` |

**None of these have real-world meaning for a software trading daemon.** Pain, oxygen, toxins, and stability are fictional body metaphors. When they trigger, they silently override whatever market intelligence the agent was about to act on.

**Verdict:** HARMFUL.

**Action: Make `get_reflex_override()` always return `None`. The method exists and is called; neutralizing it here stops the entire reflex chain without touching the callers. One-line fix per condition.**

---

### 3. InhibitionSystem — Low-Energy NoGo Branch
**Files:** `mae_core/coordination/inhibition_system.py` lines 101–104, `mae_core/agents/lifecycle_inhibit_decide.py` lines 18–82
**What it is:** Basal ganglia Go/No-Go gate. Runs on every agent step and can suppress action.

**How it harms:** Three of its five NoGo sources are fake-physiology-fed:
- **Low energy** (`energy_level < 0.2`, line 102): feeds from the permanently-drained EnergyReserve described above. With reserves near zero, this adds 0.24–0.3 NoGo pressure every step.
- **Somatic warning** (`emotional_valence < -0.3 and arousal > 0.6`, line 107): EmotionalSystem maps high-cortisol + adrenaline to FEAR (valence -0.8, arousal 0.9). Market stress events that legitimately raise cortisol will produce FEAR emotion → somatic NoGo.
- **Surprise brake** (`prediction_error > 0.6`, line 92): High prediction error produces NoGo. Agents in novel market conditions (exactly when you want them active) are most likely to have high prediction error.

The Go pathway's baseline is 0.2 + 0.3×goal_priority (typically 0.35). A permanently low energy level adds 0.27 NoGo, which alone can cross the 0.15 threshold and suppress action.

**Note:** The inhibition system also has a good property — the safety override at line 134 forces disinhibition after 5 consecutive inhibitions. And the high-risk branch (line 99) is actually useful (risk_score > 0.7 = danger). But the low-energy and somatic branches are fed by fictional sensors.

**Verdict:** HARMFUL in its current wiring (energy branch + somatic branch under normal operation).

**Action: Remove the energy_level NoGo branch (lines 101–104 of `inhibition_system.py`). Keep the risk_score and prediction_error branches — those are market-relevant. Consider raising the somatic valence threshold from -0.3 to -0.7 so only FEAR-level emotions inhibit (not mild negativity).**

---

### 4. CircadianRhythm — Sensing Worker Throttle
**Files:** `mae_core/coordination/circadian_rhythm.py`, `mae_core/market/sensing_hook.py` line 317, `mae_core/market/sensing_scheduler.py` line 50
**What it is:** A step-count-based phase cycle (ACTIVE 60 steps → CONSOLIDATION 25 steps → REST 15 steps of every 100-step cycle).

**How it harms:** During CONSOLIDATION phase, `get_activity_multiplier()` returns 0.5. During REST it returns 0.1. These multipliers are passed to the sensing hook via `set_circadian_scale()` and applied directly at `sensing_scheduler.py` line 50:

```python
_max_concurrent = max(3, int(12 * getattr(self, "_circadian_scale", 1.0)))
```

This means:
- ACTIVE: 12 concurrent sensing workers
- CONSOLIDATION: 6 workers (50% capacity)
- REST: 3 workers (25% capacity — the floor)

A 100-step cycle = 15 steps at 25% capacity. For a 24/7 trading daemon operating at pace 2.0, that's 15 half-second windows with 75% of sensing capacity disabled, repeating endlessly. Markets do not pause for simulated sleep. Earnings calls, Fed announcements, and flash crashes don't care about MIDGE's REST phase.

**The CONSOLIDATION phase already has a legitimate positive use:** it triggers `hypothesis_engine.step()` and `excavation_daemon.step()` via `_wire_memory_consolidator`. That's good. The harm is exclusively from the worker count reduction.

**Verdict:** HARMFUL (the sensing throttle). The phase-triggered jobs are beneficial.

**Action: Decouple sensing worker count from circadian phase. Fix `_circadian_activity = 1.0` permanently in `_wire_circadian` so the multiplier never changes. Keep the phase-change callbacks that trigger hypothesis and excavation jobs — those are doing real work.**

---

### 5. Cortisol → ResourceGovernor Budget Tightening
**Files:** `mae_core/coordination/endocrine_consumers.py` lines 176–198, `mae_core/market/resource_governor.py` lines 195–216
**What it is:** Endocrine wiring that calls `resource_governor.tighten_budgets(level)` when cortisol > 0.6, reducing EXPLORE-tier API source hourly budgets by the cortisol factor.

**How it harms:** Cortisol rises from market stress events (deception detected, threat detected, failed predictions). These are exactly the moments when MIDGE should be fetching MORE data, not less. The biological instinct (conserve energy under stress) is the opposite of what a trading intelligence system needs under market stress. High volatility is when the most signals matter.

Additionally, `tighten_budgets(level)` uses the raw cortisol level (0.6–1.0) as the multiplication factor — meaning at cortisol=0.7, budgets are reduced to 70% of their limit. At cortisol=0.9 (common during rapid market moves), budgets drop to 90% of the already-tightened limit, compounding over time. There is no automatic re-expansion until cortisol drops below 0.3.

**Verdict:** HARMFUL. Stress is the wrong trigger for data reduction in a trading system.

**Action: Unregister the cortisol consumer for ResourceGovernor. The ResourceGovernor itself (hourly rate limits) remains valuable — those are real API rate limits. But the cortisol coupling should be removed. In `bio_market_wiring.py`, do not call `endocrine.register_resource_governor(ctx.resource_governor)`.**

---

### 6. RespiratorySystem — Oxygen Drain from Market Signals
**Files:** `mae_core/bootstrap/bio_market_wiring_extended_b.py` lines 85–101, `mae_core/coordination/organism_state_outputs.py` line 77
**What it is:** RespiratorySystem tracks "oxygen" (0–1) as processing capacity. Market wiring drains it: convergence alerts consume 0.03 oxygen each; velocity anomalies consume up to 0.1.

**How it harms:** When oxygen drops below 0.3, OrganismState publishes `coordination.hypoxia`, which flows to `_on_respiration()` (subscriptions.py line 201), setting `_oxygen_level < 0.3`. Then `get_reflex_override()` at line 77 returns `"rest"`.

The convergence alerter runs continuously. Busy market sessions with multiple convergence alerts per cycle will drain oxygen faster than the natural replenishment rate (0.1/step baseline). This creates a treadmill: the more MIDGE succeeds at her primary job (detecting convergences), the more she throttles herself.

**Verdict:** HARMFUL — the oxygen drain chain is a success-punishment loop.

**Action: Remove the convergence and velocity callbacks that call `respiratory.consume_oxygen()`. The RespiratorySystem itself can remain as a monitoring tool; it just should not drain based on market signal volume. Alternatively, if `get_reflex_override()` is fully neutralized (Recommendation 2), the oxygen branch becomes harmless even with draining.**

---

### 7. _decide() Collision Avoidance — Danger Gradient Rest
**File:** `mae_core/agents/lifecycle_inhibit_decide.py` lines 107–123
**What it is:** Two checks early in the `_decide()` cascade that return `"rest"` based on collision risks and danger gradients from the stigmergic environment.

**How it harms:**
```python
if collision_risks:
    if len(collision_risks) > 0:
        return "rest"  # Stop to avoid collision

if danger_strength > 0.5:
    return "rest"
```

These checks run before any market intelligence logic. The collision detection is a spatial agent model concept (agents avoiding each other in a grid environment). MIDGE's agents are not navigating physical space. `_collision_risks` and `_danger_gradient` are vestigial navigation metaphors. If either is set to any non-empty/non-zero value by any system, every agent is frozen.

**Verdict:** HARMFUL — these are physics-simulation leftovers that have no meaning in MIDGE's context.

**Action: Remove both collision avoidance blocks from `_decide()`. They have no market intelligence equivalent.**

---

## NEUTRAL SYSTEMS — No Active Harm, No Value

---

### 8. NociceptionSystem
**File:** `mae_core/communication/nociception.py`
**What it is:** Accumulates "pain" from system failures and threats. Publishes total pain load.

**Current state:** The pain is consumed by OrganismState (`_pain_load`) and was the input to `get_reflex_override()` line 69 (`pain_load > 0.8`). If that reflex is neutralized (Recommendation 2), NociceptionSystem becomes an observer with no harmful downstream effects. It does publish to EventBus, which InhibitionSystem reads via the somatic marker pathway, but only if valence and arousal are both high simultaneously.

**Verdict:** NEUTRAL after reflex neutralization. Produces noise on EventBus but causes no direct suppression.

**Action: KEEP as-is if reflex_override is neutralized. Could be repurposed as a market health tracker (pain = failed API calls, chronic pain = persistently broken data sources).**

---

### 9. HomeostasisRegulator
**File:** `mae_core/coordination/homeostasis.py`
**What it is:** Monitors 7 parameters against setpoints and publishes correction signals.

**Current state:** The corrections are consumed by `_on_homeostasis_correction` in OrganismState (subscriptions.py line 171–177), which sets `_homeostasis_deviation`. This feeds the `homeostasis_deviation >= 0.7` reflex in `get_reflex_override()`. If that reflex is neutralized, HomeostasisRegulator produces monitoring data without causing any suppression.

**Verdict:** NEUTRAL after reflex neutralization. The setpoints (energy, cortisol, dopamine, serotonin) are fictional soft metrics with no real-world meaning. The corrections are advisory signals that go nowhere impactful once the reflex chain is broken.

**Action: KEEP as monitoring infrastructure. Could be repurposed to track real market health metrics (signal quality, convergence hit rate, API error rate) against setpoints.**

---

### 10. VestibularSystem
**File:** `mae_core/coordination/vestibular_system.py`
**What it is:** Tracks metric variance across a rolling window and computes a stability score.

**Current state:** Publishes stability score to `coordination.balance_update`. OrganismState reads it into `_stability`. This feeds the `stability < 0.3` reflex. If reflex is neutralized, VestibularSystem is a variance monitor.

**Verdict:** NEUTRAL after reflex neutralization. Stability tracking of real market volatility metrics could actually be useful.

**Action: KEEP. Consider repurposing: have it track actual market signal volatility (Thompson score variance, convergence rate variance) as a genuine instability detector.**

---

### 11. DigestiveSystem
**File:** `mae_core/coordination/digestive_system.py`
**What it is:** Queues and prioritizes "nutrient packets" with an energy budget.

**Current state:** No market data flows through it. It operates on abstract internal packets that no production code feeds. Its energy budget drains independently from EnergyReserve. Its output (`coordination.digestion_complete`) sets `_digestion_active` in OrganismState — a metric that's tracked but not used in any decision.

**Verdict:** NEUTRAL — it's an island with no connections to market intelligence.

**Action: KEEP as architectural structure (satisfies triadic constraints). No harm, no benefit.**

---

### 12. SenescenceManager
**File:** `mae_core/emergent/senescence.py`
**What it is:** Tracks wear/aging per system, publishes rejuvenation requests.

**Current state:** OrganismState tracks `_organism_age` from age updates. No decision paths use organism age. No reflex override based on age.

**Verdict:** NEUTRAL. Could be useful as a "signal source reliability aging" tracker (sources that haven't produced winning signals age out), but currently does nothing harmful or helpful.

**Action: KEEP. Low-priority repurposing candidate.**

---

## BENEFICIAL SYSTEMS — Keep As-Is

---

### 13. ResourceGovernor (Base Function)
**File:** `mae_core/market/resource_governor.py`
**What it is:** Tracks real API call counts per source with hourly rate limits.

**Why it's beneficial:** These are actual external API constraints, not fictional biology. SEC EDGAR rate limits, Finnhub limits, and RapidAPI quotas are real. The ResourceGovernor correctly enforces them, prevents bans, and provides usage statistics.

**The harmful part** is the cortisol coupling (Recommendation 5). The governor itself is sound.

**Verdict:** BENEFICIAL (base function). Remove only the cortisol wiring.

---

### 14. BoundaryMembrane (API Gateway Use)
**File:** `mae_core/defense/boundary_membrane.py`, used in `mae_core/external/api_gateway.py`
**What it is:** Source trust registry — pre-registered external providers pass immediately, unknown sources are quarantined.

**Why it's beneficial:** The bootstrap pre-registers all market data providers as trusted (trust=0.8). This means SEC EDGAR, Finnhub, yfinance, etc. all pass on the fast path. Unknown sources are quarantined, which is a reasonable security posture for a system that might receive adversarial inputs.

**Caveat:** The quarantine auto-reject after 50 steps (line 311–317 of `boundary_membrane.py`) could be a problem if any new data source is added but not pre-registered in bootstrap. Check that all 31 sources are pre-registered.

**Verdict:** BENEFICIAL for API trust management.

---

### 15. ThreatDetector (Deception Detection)
**File:** `mae_core/defense/threat_detector.py`
**What it is:** Multi-strategy threat detection system.

**Why it's beneficial:** In the market wiring context, ThreatDetector is wired to deception events (`CH_DECEPTION_DETECTED`). Market deception detection (wash trading, coordinated pumps, insider obfuscation patterns) is a legitimate and valuable capability for a trading intelligence system.

**Caveat:** ThreatDetector publishes to `defense.threat_detected`, which NociceptionSystem converts to pain, which feeds pain_load, which (if reflex_override is not neutralized) causes agent freeze. After reflex neutralization, this chain is broken and ThreatDetector's threat detection value stands alone.

**Verdict:** BENEFICIAL after reflex neutralization.

---

### 16. QuorumSensing (Market Consensus)
**File:** `mae_core/communication/quorum_sensor.py`, `quorum_space.py`
**What it is:** Collective signal concentration tracking. Agents deposit signals; concentration rises; a quorum threshold signals consensus.

**Why it's beneficial:** This is legitimately repurposed for MIDGE. The market wiring (`_wire_quorum` in `bio_market_wiring_b.py`) deposits convergence alerts and pattern stack signals per ticker, and dual-confirmations add strength. High concentration on a ticker = multiple independent systems independently converging. This is genuine multi-agent consensus.

**Note:** The quorum_pressure feeds InhibitionSystem's Go pathway (a social pressure signal to act). This is one of the few bio-market wiring connections that is correctly directionally aligned — market consensus should encourage action, not suppress it.

**Verdict:** BENEFICIAL. Keep and expand.

---

## Harm Chain Diagram

```
EnergyReserve drains to 0
    → CH_STARVATION fires every step
    → OrganismState._energy_critical = True
    → OrganismState._energy_level ≈ 0
    → get_reflex_override() returns "rest" [NEUTRALIZED but still flows to:]
    → InhibitionSystem energy NoGo branch (+0.27 NoGo per step)
    → Agent action SUPPRESSED

RespiratorySystem drains on each convergence
    → CH_HYPOXIA fires
    → OrganismState._oxygen_level < 0.3
    → get_reflex_override() returns "rest" [HARMFUL]
    → Agent action SUPPRESSED

NociceptionSystem fires on threat/heal events
    → OrganismState._pain_load
    → get_reflex_override() at pain > 0.8 returns "rest" [HARMFUL]
    → InhibitionSystem somatic NoGo (if FEAR emotion active)
    → Agent action SUPPRESSED

HomeostasisRegulator fires corrections
    → OrganismState._homeostasis_deviation
    → get_reflex_override() at deviation >= 0.7 returns "rest" [HARMFUL]
    → Agent action SUPPRESSED

CircadianRhythm enters REST/CONSOLIDATION
    → sensing_scheduler._circadian_scale drops to 0.1
    → max_concurrent workers = max(3, int(12 * 0.1)) = 3
    → 75% of sensing capacity DISABLED [HARMFUL]

EndocrineSystem cortisol rises (market stress event)
    → ResourceGovernor.tighten_budgets(cortisol_level)
    → EXPLORE-tier source hourly limits reduced by 10–40%
    → Fewer API calls permitted during exactly the moments needing more data [HARMFUL]
```

---

## Recommended Fix Order

| Priority | System | Change | File | Lines |
|----------|--------|--------|------|-------|
| P1 | OrganismState reflex_override | Return None for ALL conditions (pain, stability, oxygen, toxin, homeostasis) | `organism_state_outputs.py` | 69–96 |
| P1 | `_decide()` collision blocks | Remove both collision/danger-gradient `return "rest"` blocks | `lifecycle_inhibit_decide.py` | 107–123 |
| P2 | InhibitionSystem energy branch | Remove the `energy_level < 0.2` NoGo block | `inhibition_system.py` | 101–104 |
| P2 | CircadianRhythm sensing scale | Pin `ctx._circadian_activity = 1.0` permanently in `_wire_circadian` | `bio_market_wiring_b.py` | 98–99 |
| P3 | Cortisol→ResourceGovernor | Remove `endocrine.register_resource_governor()` call | `bio_market_wiring.py` | 103–109 |
| P3 | RespiratorySystem drain | Remove convergence/velocity callbacks calling `consume_oxygen()` | `bio_market_wiring_extended_b.py` | 85–101 |
| P3 | EnergyReserve metabolism | Pin `_reserves = 100.0`, make `is_critical()` always return `False`, suppress `CH_STARVATION` | `energy_reserve.py` | 169–171, 187–197 |

---

## Systems Not Audited (Confirmed No Harm Pathway)

| System | Why No Harm |
|--------|-------------|
| AutoHealer | Heals failing systems; no market decision gates |
| LymphaticSystem | Garbage collection; no decision gates |
| Microbiome | Tracks diversity; no decision gates |
| EmotionalSystem | Publishes emotions; feeds InhibitionSystem only via somatic branch (already flagged) |
| ArousalRegulator | Adjusts hormone levels; advisory, not blocking |
| MemoryConsolidator | Memory learning; no blocking gates |
| EpisodicMemory/WorkingMemory | Pattern storage; no blocking gates |
| CuriosityDrive | Computes exploration bonuses; advisory, not blocking |
| DecisionRouter | Routes decisions; advisory cascade, falls through gracefully |
| PearlDefense | Evaluates quarantined inputs; does not block live market data (pre-registered sources bypass) |
| RenalFilter | Classifies data toxicity; feeds toxin_load → reflex (covered in P1 fix) |
| TheoryOfMind/Metacognition | Social/self awareness; informational only |
| HAVEN | Market deception flag tracker; informational only |
