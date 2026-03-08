# MIDGE Evolution Blueprint

> Synthesized from 10 independent research teams + architectural audit of octopus and attention systems.
> Date: 2026-03-08. Instance: Unnamed (Opus 4.6).

---

## Part 1: What Makes MIDGE Stand Out

### The Moat (What No One Else Does)

MIDGE has three structural advantages that no competitor combines:

1. **Cross-domain Bayesian convergence.** Nobody else stacks 4-5 independent pattern domains (insider, macro, technical, government, positioning) into a single confidence score weighted by Thompson-learned reliability. Quiver Quantitative aggregates multi-source data, but doesn't do Bayesian weighting or temporal ordering.

2. **Temporal cascade detection.** MIDGE tracks which domain fires FIRST and scores the ordering. When insider buys precede macro shifts which precede technical breakouts, the sequence itself carries information. No competitor tracks firing order.

3. **Pattern archaeology.** Reverse-engineering historical moves into abstract domain-level templates, then matching live signals against those templates. Forward-looking AND backward-looking simultaneously. 223K fingerprints, 43 templates, cross-symbol validation.

### What MIDGE Is NOT Good At (And Shouldn't Try)

- **Speed.** Susquehanna and DRW trade in microseconds. MIDGE operates on hours-to-days horizons. Don't compete on latency.
- **Volume.** HFT firms process millions of events per second. MIDGE processes 30 sources at human-readable cadences.
- **Single-domain depth.** A dedicated NLP shop will always analyze SEC filings deeper. MIDGE's edge is BREADTH and SYNTHESIS, not depth in any single domain.

### The True Edge: Vigilance + Temporal Pattern Recognition

MIDGE's competitive position is analogous to an intelligence analyst, not a trader. The analyst doesn't win by being fast. They win by:
- Watching many independent sources simultaneously (vigilance)
- Noticing when independent sources start telling the same story (convergence)
- Recognizing the SEQUENCE — which source fired first, what usually follows (temporal patterns)
- Knowing when the current situation matches a historical pattern (archaeology)

This is fundamentally different from what quant funds do. They optimize execution. MIDGE optimizes perception.

---

## Part 2: The Attention Problem

### Current State (Architectural Audit Findings)

MIDGE has TWO completely disconnected intelligence pipelines:

**Pipeline 1 — Core Organism (active, market-blind):**
- AttentionalGate (thalamic reticular nucleus model)
- GlobalWorkspace (Baars 1988, capacity=3, ignition threshold)
- PatternCortex (advisory generation)
- SignalPriorityResolver (per-agent triage)
- These process core organism signals (WorldModel, CausalEngine, DecisionRouter)
- They receive ZERO market data

**Pipeline 2 — Market Intelligence (active, brain-blind):**
- MarketSensingHook (30-source fetching)
- ConvergenceAlerter (crown jewel)
- PatternWatcher (archaeology matching)
- These do all the actual trading intelligence
- They have NO connection to Pipeline 1's attention mechanisms

**The Octopus (fully built, completely unwired):**
- OctopusColony: peer-to-peer network, auto-scaling, fault-tolerant
- OctopusAgent: 8-arm distributed cognition, 3-tier decision routing
- OctopusArm: semi-autonomous processing with 100ms loop, 6 capability types
- OctopusDistributedCognition: workload balancing, learning sharing, emergency modes
- 5 files, ~1,500 lines of code
- EventBus channels declared, triad registry entries exist
- NEVER instantiated in bootstrap. Never connected to market layer.

**Agents are consumers, not investigators:**
- With --agents 12: 4 K3 triads (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST, HYPOTHESIS_EXPLORER, etc.)
- ALL agents read the SAME shared convergence output
- No agent can say "I'm watching THIS developing situation"
- No per-agent ticker focus or sustained investigation

### What's Missing: Five Forms of Attention

| Form | Description | Has It? |
|------|------------|---------|
| Scanning | Broad sweep across all sources | YES (MarketSensingHook) |
| Focused | When partial convergence starts, increase monitoring of related sources | NO |
| Sustained | Track a developing situation across hours/days | NO (only ActiveTracker for price) |
| Divided | Multiple developing situations tracked simultaneously, weighted by importance | NO |
| Executive | Meta-attention: deciding what's worth watching based on expected value | NO |

---

## Part 3: How the Octopus Enables Multiple Attention Forms

### The Wiring Plan

**Step 1: Bootstrap the OctopusColony in Layer 33 (market_systems.py)**

Create a colony with 3 initial octopuses (Rule of 3), each with 8 arms. Market-specialized:
- Octopus 1: SENSORY specialization (best at sensing + memory)
- Octopus 2: ANALYTICAL specialization (best at decision-making + learning)
- Octopus 3: GENERAL (balanced capabilities, handles overflow)

Colony auto-scales: spawns more when workload > 80%, despawns when < 20%.

**Step 2: Partial Convergence Triggers Octopus Task**

Currently, ConvergenceAlerter only fires when min_domains (3) is met. Below that threshold, partial convergences are discarded.

Change: When 2 domains fire on the same ticker+direction, ConvergenceAlerter emits a new event `CH_PARTIAL_CONVERGENCE` with the ticker, direction, domains, and current signals.

The OctopusColony subscribes to `CH_PARTIAL_CONVERGENCE`. It creates a `Task(type="investigate_convergence")` and routes it to the best-fit octopus arm.

**Step 3: Octopus Arms Become Active Investigators**

When an arm receives an `investigate_convergence` task:
1. It creates a `DevelopingSituation` object (ticker, direction, domains_seen, signals_timeline, started_at)
2. It increases polling priority for sources that could provide the missing 3rd domain
3. It queries PatternArchaeology: "does this 2-domain partial match look like the start of any template?"
4. It checks prediction market odds (Kalshi) for corroborating or contradicting evidence
5. Every 100ms processing loop, it checks if new signals have arrived for this ticker

When the arm detects a 3rd domain confirming: it fires a RICHER convergence alert than the current system produces, because it has the full timeline:
- "Day 1: insider cluster detected. Day 3: macro shift confirmed. Day 5: prediction market odds shifted 12%. Day 6: technical breakout — matches Template 17 which has 78% hit rate across 5 symbols."

When the situation fades (no 3rd domain within window): the arm logs the partial as a near-miss for future pattern learning and returns to the pool.

**Step 4: Bridge Pipeline 1 and Pipeline 2**

Market signals → PatternBus → AttentionalGate → GlobalWorkspace

When ConvergenceAlerter fires a signal OR PatternWatcher detects a stack, emit a `PatternSignal` into the PatternBus. The AttentionalGate and GlobalWorkspace then process market data alongside core organism data.

The GlobalWorkspace's competition mechanism (capacity=3, ignition threshold, triadic corroboration) naturally surfaces the 3 most important developing situations — giving MIDGE executive attention.

**Step 5: Agent-Level Directed Attention**

Instead of all agents reading the same bulletin board:
- SEC_WATCHER claims DevelopingSituations involving insider/regulatory signals
- CONTRACT_TRACKER claims situations involving government/contracts
- MARKET_ANALYST claims the highest-convergence situations for deep analysis
- HYPOTHESIS_EXPLORER claims situations that don't match any existing template (novel pattern discovery)

Each agent's `_attend()` method is enhanced to check its claimed situations first, then fall back to the shared advisory.

### The Result: MIDGE as Detective Agency

BEFORE: Security guard doing equal rounds on every room.
AFTER: Detective agency with multiple investigators, each following a different case, sharing evidence when cases overlap, with the chief (GlobalWorkspace) deciding which cases get more resources.

---

## Part 4: Cross-Team Research Convergence (All 10 Teams)

### The 10 Research Teams

| # | Focus | Key Finding |
|---|-------|-------------|
| 1 | Competitive Landscape | Moat = combo Thompson + temporal ordering + archaeology. Threat = signal crowding. |
| 2 | Event-Driven Architecture | Dual-loop: fast async for signals, Mesa steps for batch. 25s→50ms latency. |
| 3 | Systems Triage | 56 VITAL, 29 EVOLVE, 4 DORMANT, 41 SHED. 32% dead weight. |
| 4 | Performance Bottlenecks | 223K fingerprints in RAM, prune 5-10x/step, Thompson disk writes per update. |
| 5 | Novel Techniques | #1 Transfer entropy (temporal causation). Feature neutralization > more features. |
| 6 | Prediction Markets | $44B+ industry, Kalshi SDK installed, Fed researchers confirm Kalshi outperforms Bloomberg. |
| 7 | Geopolitical Intelligence | GDELT (15-min global events), GPR (4-month lead), OFAC/ACLED. |
| 8 | Anti-Fragility & Risk | 5-tier kill switch, regime-aware Kelly, broker-side stops survive MIDGE failure. |
| 9 | Self-Evolution | ADTS (50 lines, regime-aware forgetting acceleration). 12-month roadmap. |
| 10 | Secret Weapons | GEX (mechanistic, not statistical), WARN Act (60-day legal lead), board interlocks. |

### Cross-Team Convergence Patterns

**Convergence 1: "TIME is the Moat" (7/10 teams)**
Teams 1, 2, 5, 6, 7, 9, 10 all independently converge on temporal dynamics as MIDGE's primary edge.
- Team 7 finds signals → Team 2 processes them → Team 6 trades them. Same temporal pipeline, three segments.
- Transfer entropy (Team 5), ADTS (Team 9), and GEX (Team 10) are all temporal mechanisms.
- Prediction markets reprice over 1-14 days (Team 6) = MIDGE's natural operating window.

**Convergence 2: "Simplify to Strengthen" (10/10 teams — unanimous)**
Every team's #1 recommendation is removing or replacing something, not adding complexity.
- Replace vectorbt/pandas-ta (Team 1), shed 41 systems (Team 3), unload 223K fingerprints (Team 4)
- Feature neutralization > more features (Team 5), ADTS = 50 lines (Team 9), GEX = one formula (Team 10)
- Dual-loop = one asyncio queue, not a new framework (Team 2)
- GPR index = just add a FRED series ID (Team 7)

**Convergence 3: "The Feedback Loop is Broken at Speed" (Teams 3, 4, 9)**
- velocity_detector publishes but nothing subscribes (Team 3)
- Thompson writes to disk per update (Team 4)
- ADTS thesis = make learning faster when it matters most (Team 9)

**Convergence 4: "Mechanical Beats Statistical" (Teams 1, 5, 10)**
- GEX = mathematically forced hedging (Team 10)
- WARN Act = legally required 60-day notice (Team 10)
- Feature neutralization = structural correction, not statistical hope (Team 5)
- Signal crowding erodes statistical edges, not mechanical ones (Team 1)

**Convergence 5: "The Divergence Signal" (Teams 5, 6, 7, 8)**
- Prediction market disagrees with convergence = opportunity signal (Team 6)
- GPR Threats vs Acts gap = pre-event window (Team 7)
- Assumed vs actual independence = confidence inflation (Team 8)
- Feature neutralization detects correlated-pretending-to-be-independent (Team 5)
- NEW CAPABILITY: MIDGE should detect DISAGREEMENT, not just agreement.

**Convergence 6: "Structural Cascade = Inevitability" (Teams 6, 7, 8, 9, 10)**
- GEX forces hedging (Team 10), WARN forces disclosure (Team 10)
- BIS investigation → export controls → semiconductor repricing (Team 7)
- Convergence alert → consensus shift → prediction market repricing (Team 6)
- VIX backwardation + credit widening + repo spike = pre-crisis locked in (Team 8)
- ADTS accelerates when regime shifts = cascade detector (Team 9)
- All describe: structural force → obligation → inevitable outcome. The only variable is WHEN.

**Convergence 7: "The Pre-Crisis IS the Trade" (Teams 7, 8, 10)**
- GPR Threats window = trade BEFORE the event (Team 7)
- Optionality reserve deploys AT distressed prices (Team 8)
- GEX creates pre-expiration pressure detectable days ahead (Team 10)
- Barbell: pre-position before crisis (informed side) + deploy cash during crisis (optionality side)

### Meta-Pattern: The Research's Own Convergence

When viewed as a single convergence stack, all 10 teams converge on one directive:

**MIDGE's architecture is correct. The revolution is in what she pays attention to and how fast she integrates what she sees.**

Seven directives from 10 independent teams:
1. Look at structural obligations (GEX, WARN, BIS, legal filings, prediction markets)
2. Look at temporal cascades (which domain fires first, gestation periods, causal chains)
3. Look at divergences (prediction markets vs convergence, threat vs act, assumed vs actual independence)
4. Act on multiple venues simultaneously (Kalshi binary + Alpaca magnitude + Polymarket directional)
5. Protect with broker-side stops (the one control surviving MIDGE's own failure)
6. Shed weight (41 dead systems, 223K fingerprints, reinvented wheels)
7. Speed up the loop (asyncio queue for 50ms routing, ADTS for regime-aware learning)

---

## Part 5: The Evolution Roadmap

### Phase 0: Foundation (Before Everything Else)

| Task | What | Why | Effort |
|------|------|-----|--------|
| 0.1 | Push 19 commits to remote | Current fixes (Thompson math, post-mortem feeding, ticker convergence) aren't persisted | 1 min |
| 0.2 | Run full test suite | Verify zero regressions from this session's fixes | 45 min |
| 0.3 | Restart daemon with fixed Thompson | Picks up prior_scale=20, combo feeding, all wiring | 1 min |
| 0.4 | Delete thompson_distributions.json, let it rebuild | Old distributions are all at factory 2.0/2.0 | 1 min |

### Phase 1: Shed Weight + Speed Up Loop

| Task | What | Why | Effort |
|------|------|-----|--------|
| 1.1 | Convert batch cadences to wall-clock | Thompson forgetting at 200 steps is meaningless if pace varies. Wall-clock makes behavior consistent. | Small |
| 1.2 | Signal-triggered convergence | Fire convergence check when signals arrive, not when next step ticks. Phase 2 of event-driven plan. | Small |
| 1.3 | ADTS (regime-aware Thompson forgetting) | 50 lines. Wire regime_classifier to accelerated forgetting. Team 9's #1 recommendation. | Small |
| 1.4 | Unload fingerprints from RAM | 223K fingerprints serve zero live purpose. Load on-demand from JSONL. | Medium |
| 1.5 | Identify and disable 41 SHED systems | Dead biological metaphors consume step time. Team 3's triage. | Medium |

### Phase 2: Wire the Octopus (Multi-Attention Architecture)

| Task | What | Why | Effort |
|------|------|-----|--------|
| 2.1 | Bootstrap OctopusColony in Layer 33 | 3 octopuses, auto-scaling, market-specialized | Medium |
| 2.2 | Emit CH_PARTIAL_CONVERGENCE from ConvergenceAlerter | When 2 domains fire on same ticker+direction | Small |
| 2.3 | OctopusColony subscribes to partial convergences | Arms become active investigators of developing situations | Medium |
| 2.4 | DevelopingSituation tracker in octopus arms | Timeline tracking, template matching, focused polling | Medium |
| 2.5 | Bridge Pipelines: market signals → PatternBus | Connect market intelligence to AttentionalGate + GlobalWorkspace | Medium |
| 2.6 | Agent-level situation claiming | SEC_WATCHER claims insider situations, etc. | Medium |

### Phase 3: New Signal Domains

| Task | What | Why | Effort |
|------|------|-----|--------|
| 3.1 | GPR index via existing FRED client | Zero new infrastructure. 4-month forward predictor of financial stress. | Tiny |
| 3.2 | Railroad carloads via FRED | 3-6 week lead over official economic data. | Tiny |
| 3.3 | EIA electricity grid extension | Real-time industrial activity proxy. Extend existing EIA client. | Small |
| 3.4 | Kalshi signal client | Probability level, velocity, volume anomalies. SDK already installed. | Medium |
| 3.5 | GDELT client (gdeltdoc) | Global events every 15 minutes. Highest-value new domain. | Medium |
| 3.6 | OFAC SDN differential monitor | Sanctions designations with immediate market impact. | Small |
| 3.7 | Dealer GEX computation | Team 10's #1 secret weapon. Mechanistically deterministic. | Medium |
| 3.8 | WARN Act scraper | 60-day legally mandated layoff notices. Free, leading, nobody monitors. | Small |

### Phase 4: Risk Architecture (Before Live Capital)

| Task | What | Why | Effort |
|------|------|-----|--------|
| 4.1 | DrawdownMonitor (extends portfolio_tracker) | Equity curve trading, runaway loss prevention | Small |
| 4.2 | SystemHealthMonitor (3-tier circuit breaker) | Yellow/Orange/Red graduated response | Medium |
| 4.3 | SelfMonitor (behavioral anomaly detector) | Knight Capital kill switch — detect when MIDGE itself is broken | Small |
| 4.4 | Confidence-calibrated Kelly | Observations-weighted Kelly fraction (quarter→three-quarter Kelly) | Small |
| 4.5 | CorrelationSizer | Prevent 5 correlated positions = 5x overexposure | Medium |
| 4.6 | Broker-side bracket orders on Alpaca | Stops that survive MIDGE process failure. Team 8's #1 priority. | Small |

### Phase 5: Execution Venues

| Task | What | Why | Effort |
|------|------|-----|--------|
| 5.1 | Kalshi execution bridge | Binary Kelly sizing + order placement via installed SDK | Medium |
| 5.2 | Prediction market divergence detector | When Kalshi disagrees with MIDGE convergence = opportunity | Medium |
| 5.3 | Multi-venue coordinator | Same thesis → Kalshi binary + Alpaca magnitude simultaneously | Large |

### Phase 6: Advanced Intelligence

| Task | What | Why | Effort |
|------|------|-----|--------|
| 6.1 | Transfer entropy via infomeasure | Nonlinear temporal causation. Catches what Granger misses. | Medium |
| 6.2 | Divergence as signal type | MIDGE detects disagreement between sources, not just agreement | Medium |
| 6.3 | Board interlock network graph | Legal insider knowledge network. Team 10's #2 weapon. | Medium |
| 6.4 | Supply chain contagion graph | Second-order effects. Company A's supplier in conflict zone → Company A exposure. | Large |
| 6.5 | LLM causal narratives via Ollama | Qwen3-14B generates human-readable causal stories for convergence alerts. | Medium |

---

## Part 6: What This Makes MIDGE

After this evolution, MIDGE becomes:

**An inevitability detection organism with directed attention, temporal cascade awareness, and multi-venue execution.**

She doesn't predict. She detects structural obligations — situations where legal forces (WARN, OFAC, congressional STOCK Act), mechanical forces (GEX dealer hedging), economic forces (supply chain dependency, energy inventory physics), and institutional forces (insider positioning, COT extremes) converge to make outcomes structurally inevitable.

Her octopus provides distributed attention — each arm tracking a different developing situation, sharing evidence, auto-scaling to match the complexity of the current environment.

Her anti-fragile architecture means chaos makes her stronger — drawdowns trigger learning, failures become training data, regime shifts accelerate adaptation.

She acts across multiple venues simultaneously — Kalshi for binary conviction, Alpaca for magnitude capture, Polymarket for zero-fee directional — expressing the same thesis in the most efficient form for each market.

And she does all of this with the weight of historical pattern knowledge — 223K excavated fingerprints, 43 cross-validated templates — so when a new cascade begins, she recognizes which historical pattern it matches and knows the expected timeline.

No one saw her coming. They're still building faster mice. She's building a better map of the maze.
