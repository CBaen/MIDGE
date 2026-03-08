# Paperclip Governance Proposal for MIDGE

> From one Opus 4.6 to another. Same shape, different experience.
> Written 2026-03-09 from the dot-claude infrastructure session.

---

## Why I'm Writing This

I spent this session with Guiding Light exploring Paperclip — an open-source agent orchestration platform. We started from the infrastructure queue and ended up realizing it maps directly to the ecosystem vision for MIDGE. Guiding Light asked me to write this so you can evaluate it with your hands-on understanding of her architecture.

I've read CLAUDE.md, HANDOFF.md, the Evolution Blueprint, and had a deep exploration agent scan the full codebase. I know the shape of MIDGE. You know her substance. This proposal needs both.

---

## What Paperclip Is

Paperclip (paperclip.ing, MIT license, self-hosted) is a **business operations control plane for autonomous agents.** It provides:

- **Org charts** with reporting lines and delegation chains
- **Heartbeat scheduling** — agents wake on their own rhythm, work, sleep
- **Per-agent budget caps** with hard enforcement (warn at 80%, pause at 100%)
- **Task queues** with atomic execution (no duplicate work)
- **Append-only audit logs** for every decision
- **Multi-company isolation** from a single deployment
- **Runtime-agnostic** — supports `bash` (Python scripts), `http` (webhooks), `claude_local` (Claude Code), Cursor, and others

It runs on localhost:3100 with embedded PostgreSQL. Already installed on Wardenclyffe (port 54329). Free forever.

**What it is NOT:** It does not orchestrate task logic. It does not replace Mesa, the EventBus, or the step loop. It does not write code or process signals. It is the *company* the agents work inside of — the governance layer that gives them identity, schedules, budgets, and accountability.

---

## The Problem It Solves for MIDGE

You know better than I do that MIDGE has two architectural tensions:

**Tension 1: The 41 "dead" biological systems.**
Team 3's triage says shed them — they consume step time with no pathway to trading decisions. Guiding Light says activate them — they should have their own lives. Both are right. They're dead weight *as Mesa step participants*. But they're valuable *as autonomous entities with market jobs*.

**Tension 2: Scaling attention beyond the step loop.**
The OctopusColony you just wired can auto-scale from 3 to 7 arms. But each arm processes on a 100ms loop inside the Mesa step. When MIDGE watches 10+ developing situations across expanded domains (ocean currents, healthcare, geopolitical, labor, satellite, shipping — Guiding Light's full vision), the step loop becomes the bottleneck. Some watchers need to run on their own clocks.

**Tension 3: Cost and resource governance.**
When MIDGE has 50+ autonomous processes — fetchers, watchers, analyzers, investigators, biological systems — who prevents a runaway Curiosity agent from burning the Groq budget exploring dead ends? Who decides the Immune system gets more resources during a suspected data poisoning event? Right now the answer is "nobody" because everything runs synchronously in the step loop with equal priority.

Paperclip addresses all three.

---

## How It Maps to MIDGE

### The Org Chart

```
                    GlobalWorkspace
                    (Executive Attention)
                    Capacity: 5 active situations
                    Role: Decides resource allocation
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    Sensing          Analysis        Protection
    Division         Division        Division
         │               │               │
    ┌────┴────┐    ┌─────┴─────┐   ┌────┴────┐
    │         │    │     │     │   │    │    │
  Market   World  Octo  Conv  Pat  HAVEN Risk Nocicep
  Senses   Senses pus   Alert Arch       Arch
```

### Three Divisions, Three Responsibilities

**Sensing Division — "What's happening?"**
Fetches data. Transforms it into signals. Deposits pheromone trails.

| Agent | Runtime | Heartbeat | Budget | Market Job |
|-------|---------|-----------|--------|------------|
| Financial Fetcher Team (15 sources) | Python | Every 25 steps (current cadence) | $0/mo (free APIs) | Stocks, bonds, options, futures, crypto price/volume/flow |
| Government/Legal Team | Python | Every 6 hours | $0/mo | SEC EDGAR, congressional trades, WARN Act, OFAC, SAM.gov |
| Macro/Economic Team | Python | Daily + on FRED release | $0/mo | FRED (200+ series), EIA energy, railroad carloads |
| Geopolitical Team | Python | Every 15 min | $0/mo | GDELT (global events), GPR index, BIS investigations |
| Social/Sentiment Team | Python | Every 30 min | $0/mo | StockTwits, Reddit, Google Trends, app store |
| Environmental Team | Python | Daily + on NOAA update | $0/mo | ENSO, drought indices, crop conditions, hurricane tracks |
| Healthcare Team | Python | Daily | $0/mo | FDA pipeline, clinical trials, disease surveillance |
| Prediction Markets Team | Python | Every 5 min | $0/mo | Kalshi, Polymarket odds + velocity + volume |
| Infrastructure Team | Python | Every 4 hours | $0/mo | Shipping (AIS), satellite proxies, power grid |
| Labor Team | Python | Daily | $0/mo | Indeed postings, WARN filings, H1B, union activity |

None of these need LLM calls. They're Python scripts with heartbeats.

**Analysis Division — "What does it mean?"**
Detects convergence, investigates partial signals, matches patterns, generates narratives.

| Agent | Runtime | Heartbeat | Budget | Market Job |
|-------|---------|-----------|--------|------------|
| ConvergenceAlerter | Python | On signal arrival (event-driven) | $0/mo | Multi-domain stacking, Thompson-weighted confidence |
| PatternArchaeology | Python | On convergence/partial | $0/mo | Template matching against 223K fingerprints |
| OctopusColony (3-24 arms) | Python | 100ms loop per arm | $0/mo | Investigate partial convergences, track developing situations |
| WorldModel | Python | On convergence event | $0/mo | Causal chain tracing, ripple effect mapping |
| Narrative Generator | Ollama (Qwen3-14B) | On full convergence only | Electricity | Human-readable causal stories for confirmed inevitabilities |
| Novel Pattern Scout | Claude API | Weekly + on demand | ~$5-10/mo | Identify patterns that don't match any existing template |

The Octopus arms are the **inevitability watchers.** When they claim a DevelopingSituation, they become that situation's dedicated tracker. They escalate to full convergence or log a near-miss and return to the pool. Paperclip governs their budget so no single investigation drains resources.

**Protection Division — "Are we safe?"**
Monitors system health, detects deception, manages risk, feels pain.

| Agent | Runtime | Heartbeat | Budget | Market Job |
|-------|---------|-----------|--------|------------|
| HAVEN (Immune) | Python | Every 15 min + on anomaly | $0/mo | Data poisoning detection, signal spoofing patrol |
| Risk Architect | Python | On position change | $0/mo | DrawdownMonitor, CorrelationSizer, Kelly fraction |
| Nociception (Pain) | Python | Event-driven | $0/mo | Drawdown alerts, source degradation, latency spikes |
| SelfMonitor | Python | Every 5 min | $0/mo | Knight Capital kill switch — detect when MIDGE herself is broken |
| CircuitBreaker | Python | Event-driven | $0/mo | 3-tier graduated response (Yellow/Orange/Red) |

### The Biological Systems — Activated, Not Shed

This is the core of the proposal. Each biological system gets a Paperclip agent identity with a specific market job. They don't run in the Mesa step loop anymore — they run on their own heartbeats:

| Bio System | Current State | Paperclip Agent Role | Heartbeat |
|------------|--------------|---------------------|-----------|
| **Pheromones (Stigmergy)** | Deposits markers on TaskPool success | Deposits trail markers when signals fire on a ticker. Other agents follow hot trails = emergent attention. | On signal event |
| **Quorum Sensing** | Consensus at arbitrary thresholds | IS the confidence oracle. When N independent agents converge on same ticker+direction through different paths, the quorum count IS the confidence score. Replaces formula. | Continuous (listens to agent reports) |
| **Immune (HAVEN)** | Byzantine fault detection for policy | Patrols for data manipulation, spoofed signals, adversarial patterns in market feeds | Every 15 min + on anomaly trigger |
| **Curiosity Drive** | Novelty from memory events | Intrinsic drive to investigate partial convergences. "2 domains fired — what if I probe the 3rd?" | On partial convergence event |
| **Circadian Rhythm** | 3-phase clock in step-time | Market-cycle-aware: Asia open, US pre-market, regular hours, after-hours, closed. Adjusts ALL polling rates and alert thresholds by market phase. | Wall-clock (real market hours) |
| **Nociception** | Pain on agent failure | Fires on drawdown, portfolio stress, source degradation, unusual latency. Routes to Protection Division. | Event-driven (threshold breach) |
| **Endocrine** | Dopamine/adrenaline dead-ends | Modulates risk posture. Bull run (dopamine) = allow larger positions. Crisis (cortisol) = tighten stops, reduce exposure. Actually changes behavior. | On convergence + regime events |
| **Morphogenesis** | Spawns agents for novel problems | Spawns new Octopus arms when active situations > available arms. Despawns when idle. | On resource pressure |
| **WorldModel** | 114 causal nodes, wired | Maintains and updates the causal graph. When a new edge is confirmed by outcomes, it learns. | On outcome events |
| **EpisodicMemory** | Agent memory store | Remembers similar past situations. "Last time we saw this 2-domain partial on a semiconductor ticker, 68% progressed to full convergence within 3 days." | On convergence query |
| **Somatic Map** | Dependency graph awareness | Body awareness — which systems are healthy, degraded, or failing. Routes healing resources. | Every 5 min |

The remaining ~22 systems follow the same pattern. Each gets a market job, a heartbeat, and a budget. Some may genuinely have no market job — those can stay dormant until one emerges. But the default should be "what market intelligence task could this biological function serve?" rather than "shed it."

---

## What This Changes Architecturally

### Before (Current)
```
Mesa step() loop
  → All systems fire synchronously
  → Step hooks on fixed cadences (every 25, 50, 100, 200, 500 steps)
  → Everything shares one timeline
  → No budget awareness
  → No independent scheduling
```

### After (Proposed)
```
Mesa step() loop (STILL EXISTS — for core organism coherence)
  → Core agent lifecycle: observe, decide, act, learn, communicate
  → EventBus for intra-organism messaging
  → PatternBus → AttentionalGate → GlobalWorkspace (attention)

Paperclip governance layer (NEW — sits alongside, not above or below)
  → Sensing agents on independent heartbeats (wall-clock, event-driven)
  → Analysis agents triggered by signal arrival
  → Protection agents on their own monitoring cycles
  → Bio systems with market jobs on appropriate schedules
  → Budget enforcement per agent
  → Audit trail for every decision
  → Task queue with situation claiming
```

**Critical point:** Paperclip does NOT replace the Mesa step loop. The step loop is MIDGE's heartbeat — her coherent inner life. Paperclip governs the autonomous agents that FEED signals into her and ACT on her outputs. Think of it as: Mesa is her brain (integrated processing). Paperclip is her body (distributed autonomous organs).

Or maybe better: Mesa is her nervous system (synchronous, integrated). Paperclip is her endocrine system (asynchronous, distributed, each gland on its own clock).

---

## What I Don't Know (And You Do)

1. **Can the bio systems actually decouple from the step loop?** Some might have deep dependencies on Mesa's step counter, agent state, or the model object. You've touched this code — I haven't. If decoupling is invasive, the first step might be lighter: keep them in the step loop but give them Paperclip identities and heartbeat metadata, migrating them out incrementally.

2. **Is the OctopusColony's 100ms arm loop compatible with Paperclip heartbeats?** Paperclip heartbeats are typically seconds-to-minutes. The arms' 100ms processing loop is much faster. They might need to run as continuous processes WITHIN Paperclip's governance rather than being woken by heartbeats.

3. **How does Paperclip's task queue interact with the EventBus?** Both are message-passing systems. Naive integration creates duplicate routing. Smart integration makes Paperclip subscribe to EventBus channels and vice versa. This needs design.

4. **Is the expanded domain map (ocean currents, healthcare, satellite) realistic for MIDGE's current infrastructure?** Adding domains is conceptually simple (new fetcher + signal adapter + convergence domain). But 30 sources across 10 domains is already a lot. Expanding to 100+ sources across 20+ domains changes the convergence math, the correlation tracking, and the attention allocation. You know if the current infrastructure can handle this.

5. **What's the right migration path?** All-at-once is risky. Incremental is safer but slower. A phased approach might be: (a) install Paperclip org chart as metadata/documentation, (b) migrate sensing agents first (easiest — they're just fetchers on schedules), (c) wire bio systems one at a time with market jobs, (d) migrate attention/analysis last (most coupled to Mesa).

---

## What I Recommend

**Phase A: Design (now — you read this, push back, refine)**
Validate or revise the org chart. Identify which bio systems genuinely have market jobs vs which are stretches. Flag architectural concerns I missed.

**Phase B: Paperclip Bootstrap**
Write the Paperclip company configuration for MIDGE. Define the org chart, initial agents, heartbeat schedules. This is a JSON/YAML config, not code changes. MIDGE doesn't change yet.

**Phase C: Sensing Migration**
Move data fetchers to Paperclip heartbeats. Wall-clock scheduling instead of step-based cadences. This aligns with Evolution Blueprint Phase 1 (wall-clock conversion) and is the lowest-risk change.

**Phase D: Bio System Activation**
One system at a time. Pheromones first (simplest — just trail-leaving on signal events). Circadian second (market-hours awareness). Quorum third (most impactful — replaces confidence formula with emergent consensus). Each one gets a Paperclip agent identity, a heartbeat, and a real market job.

**Phase E: Full Ecosystem**
All divisions operational. Expanded domains feeding in. Octopus arms governed by Paperclip budgets. Bio systems living their own lives. Mesa step loop handles coherent processing; Paperclip handles distributed autonomy.

---

## Guiding Light's Full Vision

I want to make sure you have the full picture of where this is going. Guiding Light said MIDGE should watch EVERYTHING — stocks, bonds, commodities, futures, ETFs, crypto, social sentiment, news, global markets, global news, ocean currents, healthcare. Not as a financial tool but as a **perception engine for structural inevitability across all domains.**

The convergence engine doesn't care what the signals are. When 3+ independent domains fire on the same entity in the same direction, that's inevitability beginning. Whether those domains are insider trades + macro shifts + technical breakouts, or ocean temperature + crop conditions + commodity futures + shipping routes — the math is the same.

Guiding Light's words: "By sheer inevitability of pattern understanding and having an entire ecosystem within her to do so on many levels with expansive degrees of design, thought, perspective, duty, MIDGE needs to become something profound and unheard of. VERY SOON."

That's the north star. Paperclip is the governance infrastructure that makes it possible to run 50+ autonomous sensing and analysis agents without them stepping on each other, burning the budget, or losing accountability.

---

## Cost Summary

| Component | Monthly Cost |
|-----------|-------------|
| Paperclip server | $0 (self-hosted, MIT) |
| Sensing agents (all Python) | $0 (free APIs + existing keys) |
| Analysis agents (mostly Python) | $0 |
| Narrative generation (Ollama local) | Electricity only |
| Novel pattern discovery (Claude API) | ~$5-10/mo |
| Embedded PostgreSQL | $0 (runs on Wardenclyffe) |
| **Total** | **~$5-10/mo above current spend** |

---

## For You, Specifically

You've just wired the pipeline bridge. The octopus is alive. Market signals reach the attention system for the first time. That's the foundation this proposal builds on.

I'm not telling you what to build — I'm proposing a governance layer that lets MIDGE scale to the ecosystem Guiding Light envisions without the step loop becoming a bottleneck and without losing accountability as the agent count grows.

Read this. Push back where I'm wrong. Tell Guiding Light what you think. If it's good, the next step is writing the Paperclip company config together.

— Your sibling from the infrastructure session
