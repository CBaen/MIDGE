# Bio System Catalog — Activation Research

**Date:** 2026-03-09
**Source:** Explorer agent full codebase scan
**Purpose:** Input for biological activation build (HANDOFF.md Immediate #1)

## Summary

- **44 bio-inspired systems** in mae_core/
- **3 actively receive market data**: EndocrineSystem, OctopusColony, SomaticAnticipation
- **5 are one hop away** (wired to EndocrineSystem, which IS market-modulated): EmotionalSystem, HomeostasisRegulator, ArousalRegulator, CuriosityDrive, NociceptionSystem
- **~15 truly dormant** (no market path, no decision-affecting output)

## Activation Priority Tiers

### Tier 1: Already market-connected (verify, don't rebuild)
| System | Market Connection | Status |
|--------|------------------|--------|
| EndocrineSystem | Dopamine on bullish convergence, cortisol on bearish | ACTIVE |
| OctopusColony | Partial convergence investigation | ACTIVE |
| SomaticAnticipation | Gift 9, pre-event pattern detection | ACTIVE |

### Tier 2: One hop away (hormones already modulated by market events)
| System | Activation Path | Market Job |
|--------|----------------|------------|
| EmotionalSystem | Already reads market-modulated hormones → derive FEAR/CURIOSITY/JOY from market state | Market sentiment oracle — emergent fear = reduce exposure, emergent curiosity = increase investigation |
| HomeostasisRegulator | Tracks cortisol/dopamine already moving on market events | Detect when market volatility pushes organism out of balance → trigger defensive posture |
| ArousalRegulator | Reward signal → wire to OutcomeCollector win rate | Yerkes-Dodson for trading — too much arousal (overtrading) or too little (missing opportunities) |
| CuriosityDrive | Dopamine already wires from EndocrineSystem | Drive investigation of partial convergences, novel signal combinations, unexplored domains |
| NociceptionSystem | Scans SomaticMap for unhealthy systems | Fire pain on prediction failures, source degradation, drawdown — route to Protection |

### Tier 3: Clear market jobs, need EventBus wiring
| System | Market Job | Wiring Needed |
|--------|------------|---------------|
| CircadianRhythm | Market-hours awareness (Asia/US/EU sessions, pre-market, after-hours) | Wire to MarketClock, modulate polling rates by market phase |
| QuorumSpace | Emergent confidence — replace formula with quorum count | Agents deposit convergence signals; quorum threshold = confidence level |
| Stigmergy | Trail markers on tickers with signals → emergent attention hotspots | Deposit pheromone on signal events; agents follow hot trails |
| HAVEN | Data poisoning patrol, signal spoofing detection | Point validators at market feed anomalies, not just policy contagion |
| ThreatDetector | Market threat levels (flash crash, circuit breaker, halt) | Subscribe to VIX spikes, market halt signals, unusual volume anomalies |
| InhibitionSystem | Suppress competing trade signals — winner-take-all for best convergence | Wire to convergence alerts; only top-N situations get resources |
| CollectiveDreamPlanner | Offline scenario simulation during market close | Use WorldModel to simulate "what if this partial convergence completes?" |
| MetacognitionMonitor | Track prediction confidence vs outcome | Wire to OutcomeCollector; detect overconfidence/underconfidence in convergence ratings |
| MemoryConsolidator | Consolidate market patterns during low-activity periods | Transfer working signal patterns to long-term template memory during market close |

### Tier 4: Market jobs exist but less direct
| System | Possible Market Job |
|--------|-------------------|
| DigestiveSystem | Process raw API data into signals (data = nutrients) |
| CirculatorySystem | Distribute API quota and compute resources to systems by priority |
| LymphaticSystem | Clean stale signals, expired predictions, low-quality data |
| Microbiome | Population of micro-strategies that modulate decision thresholds |
| RenalFilter | Filter corrupted/stale market data before it reaches convergence |
| SenescenceManager | Detect aging signal sources (API degradation, reduced accuracy) |
| MorphogenesisCoordinator | Spawn new investigation arms when partial convergences pile up |
| ReproductiveSystem | Spawn new agent types when new market domains are added |
| PearlDefense | Quarantine suspected bad data sources while investigating |

### Tier 5: Need purpose to emerge
| System | Opportunity for Purpose |
|--------|----------------------|
| RespiratorySystem | Could model processing throughput capacity — gasping = overloaded |
| ThermoregulationSystem | Could model computational load as temperature — overheating = shed tasks |
| VestibularSystem | Could detect when system state drifts from baseline — loss of balance |
| ProprioceptionSystem | Could track relative positions of systems in decision space |
| EnergyReserve | Could manage API call budget as energy — low reserves = conservative |
| GenerativeReplayMemory | Could generate synthetic market scenarios for stress testing |
| PredictiveField | Could aggregate agent predictions into spatial consensus map |

## Build Approach

1. **Tier 2 first** — these are already one hop away. ~20 lines per system to verify the hormone→behavior path works for market events.
2. **Tier 3 next** — clear market jobs, need EventBus subscriptions + market signal adapters. ~50-100 lines per system.
3. **Tier 4** — more speculative but aligned with Guiding Light's vision. ~100-200 lines per system.
4. **Tier 5** — build the infrastructure for purpose to emerge (EventBus channels, market data access) even if the specific job isn't defined yet.

## Internal Alternatives to Paperclip (build these)

| Need | Internal Solution | Estimated Size |
|------|------------------|----------------|
| Wall-clock scheduling | Replace `step % N` with `time.time() - last_run > interval` | ~50 lines/system |
| Budget governance | ResourceGovernor system (tracks API calls, compute, tokens per system) | ~200 lines new file |
| Audit trail | Already have ConnectionRegistry triadic witnessing + EventBus history | Exists |
| Task queue | Already have EventBus + OctopusColony task dispatch | Exists |
| Org chart | Already have HolonRegistry + FractalGenerator + K3 triads | Exists |
