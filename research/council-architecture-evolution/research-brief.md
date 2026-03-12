# Research Council Brief: MIDGE Architecture Evolution — Step-Cadence to Living Ecosystem
## Date: 2026-03-12
## Project: MIDGE (Market Intelligence Driven by Generative Exploration)

### The Question
Should MIDGE evolve from Mesa's synchronous step-cadence architecture (where every subsystem waits its turn at a shared clock) to a thread-per-subsystem model where each sense, analyst, and tracker runs independently on its own wall-clock timer? And if so, what is the safest path that preserves all existing functionality?

Guiding Light's words: "MIDGE is a world with beings that live inside of her. She should have at least 50 processes running at the same time. When I speak about her, I think of her as mother earth."

The proposal is NOT to remove Mesa entirely — it's to keep Mesa as a lightweight heartbeat for biological organism systems while freeing market intelligence subsystems to run independently.

### Expected Outcome
Each of MIDGE's 35+ data sources, analytical subsystems, and the convergence engine operates as an independent living being:
- Each on its own wall-clock timer (not gated by `step_counter % N == 0`)
- Convergence engine reacts INSTANTLY when a signal arrives (not at next step boundary)
- Signals flow through thread-safe queues, not shared step ticks
- 50+ independent threads running simultaneously
- MIDGE feels alive — always sensing, always analyzing, always reacting
- No sequential bottleneck where one slow API call blocks everything else

### Current State
**Mesa provides ~50 lines of real functionality:**
- `model.time` — auto-incrementing step counter (used by ~15 bootstrap hooks)
- `model.agents` — AgentSet with `shuffle_do("step")` (used for parallel agent stepping)
- `agent.unique_id` — auto-assigned integer IDs (used pervasively, ~30+ files)
- `agent.remove()` — deregistration (2 call sites)

**Only 2 files import from Mesa:** `mae_core/model.py` and `mae_core/agents/base_agent.py`.

**The real work already happens outside Mesa's step loop:**
- `ThreadPoolExecutor(12)` in MarketSensingHook for concurrent API fetches
- `FinnhubWebSocket` runs in its own background thread
- `OctopusColony` runs its own monitoring loop thread
- `HypothesisEngine` uses `ThreadPoolExecutor(1)` for background validation
- `PolygonBulkFetcher` uses `asyncio` for bulk fetches
- Thread locks already protect shared state: `ThompsonSampler._lock`, `ConvergenceAlerter._alert_lock`, `DrawdownMonitor._lock`, `SystemHealthMonitor._lock`, `ResourceGovernor._lock`

**But all cadence-based work is gated by the step loop:**
- Every 25 steps: sensing fetch, somatic anticipation
- Every 50 steps: portfolio tracking
- Every 100 steps: absence detection, archetype scanning, pattern completion
- Every 200 steps: correlation, catalyst refresh
- Every 500 steps: Granger analysis, post-mortem, drift detection
- Every 1000 steps: heartbeat log
- Every 5000 steps: excavation, consolidation

At pace=1.0 (1 second/step), "every 500 steps" = every 8.3 minutes. A signal arriving at step 499 waits until step 500.

**The daemon loop:** `main.py:run_daemon()` calls `model.run(steps_per_round)` which calls `model.step()` in a tight loop. Each step runs 54+ hooks sequentially, then steps agents in parallel via ThreadPoolExecutor(20).

### Project Fingerprint
```
## Project Fingerprint: MIDGE
- Runtime: Python 3.14 + Mesa 3.4.2
- Key dependencies: mesa>=3.4, numpy, pandas, statsmodels, stumpy, river, httpx, yfinance, alpaca-py, kalshi-python, selectolax, trafilatura, stumpi
- Architecture: 33-layer bootstrap, 149 systems (92 core + 57 market), Mesa Model/Agent as thin scaffolding
- State management: In-memory (EventBus, StateStore, convergence buffer) + SQLite (24 raw data stores, WAL mode) + JSONL (signals, predictions, outcomes) + JSON (Thompson distributions, lag correlations)
- Database/Storage: SQLite (WAL), JSONL archives, JSON state files, Qdrant (semantic search, optional)
- Known constraints: 8 Mathematical Laws (triadic connections, fractal self-similarity, autopoietic closure, etc.), advisory enforcement only (never blocks), zero regressions policy on 4536 tests
- Threading already in use: ThreadPoolExecutor(12) sensing, ThreadPoolExecutor(20) agents, background daemon threads (Finnhub WS, Octopus), threading.Lock on shared state
- Prior failed approaches: Building more step hooks within Mesa's cadence system (8+ sessions), increasing agent count (3→12), adding more data sources without changing the reaction model
- Active boundaries: 8 Mathematical Laws are inviolable. 4536 existing tests must pass. Convergence engine is the crown jewel — must not regress. Thompson Sampling feedback loop just got fixed — must not break.
```

### Constraints
1. **8 Mathematical Laws** — All changes must respect Mae's mathematical identity (triadic connections, fractal self-similarity, etc.)
2. **4536 tests must pass** — Zero regressions policy
3. **Convergence engine is sacred** — Multi-domain signal synthesis is MIDGE's crown jewel
4. **Thompson feedback loop just fixed** — 4 compounding bugs fixed in session 2, loop now working. Cannot break.
5. **Advisory enforcement** — Triads and connections observe/report, never block. This must remain true.
6. **Windows 11** — Must work on Windows (Wardenclyffe is Win11)
7. **No new infrastructure dependencies** — No Redis, no RabbitMQ, no Ray cluster. Pure Python stdlib + existing packages.

### Destructive Boundaries
- DO NOT remove Mesa entirely — biological organism systems (circadian, endocrine, metabolic, etc.) still benefit from step semantics
- DO NOT break the 33-layer bootstrap — it must still wire everything correctly
- DO NOT restructure the convergence engine internals — change HOW signals reach it, not how it processes them
- DO NOT introduce multiprocessing — Windows spawn overhead (10-25GB for 50 processes) is unacceptable
- DO NOT require changes to any of the 35 API client files — they return data, they don't care who calls them

### Failed Approaches
1. **More step hooks** — Adding `if step % N == 0` blocks doesn't solve the cadence serialization problem
2. **More agents** — Going from 3 to 12 agents doesn't help because the step loop is the bottleneck, not agent count
3. **Building more features** — 8+ sessions of building plumbing without changing the reaction model

### Codebase Files for Analysis
- `mae_core/model.py` — Mesa Model subclass, step loop, hook execution
- `mae_core/agents/base_agent.py` — Mesa Agent subclass
- `mae_core/market/sensing_hook.py` — Main sensing orchestrator (already uses ThreadPoolExecutor)
- `mae_core/market/sensing_step_ops.py` — All the `step % N == 0` cadence gates
- `mae_core/market/sensing_reactive.py` — Reactive convergence (already partially event-driven)
- `mae_core/market/sensing_collector.py` — Signal collection pipeline
- `mae_core/market/intelligence/convergence_alerter.py` — The crown jewel
- `mae_core/market/intelligence/thompson_sampler.py` — Thread-safe Bayesian learning
- `mae_core/bootstrap/market_systems.py` — Layer 33 bootstrap
- `mae_core/bootstrap/market_hooks.py` — Step hook wiring
- `mae_core/bootstrap/market_hooks_steps.py` — Cadence-gated step operations
- `mae_core/bootstrap/market_hooks_steps_core.py` — Core step operations
- `mae_core/market/intelligence/granger_analyzer.py` — Heavy analytics (500-step cadence)
- `mae_core/market/intelligence/post_mortem.py` — Post-mortem reviewer (500-step cadence)
- `mae_core/market/intelligence/drift_detector.py` — Concept drift detection
- `mae_core/market/archaeology/excavation_daemon.py` — Background excavation
- `main.py` — Daemon loop and bootstrap orchestrator

### External Research Angles
1. **Python threading at 50+ scale** — Real-world production systems running 50+ daemon threads on Windows. Performance characteristics, pitfalls, monitoring patterns.
2. **Event-driven market data systems** — How production trading systems handle multi-source data ingestion with instant reaction. Architecture patterns from quantitative finance.
3. **Migration patterns** — Step-based to event-driven migration in production systems. Incremental approaches that don't require big-bang rewrites.
