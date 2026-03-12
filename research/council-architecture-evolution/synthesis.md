# Council Synthesis: MIDGE Architecture Evolution
## Date: 2026-03-12
## Vetted by: Orchestrator
## Council: Codebase Analyst + Devil's Advocate (External Researcher still running)

### Master Score Table

| Dimension | Codebase Analyst | Devil's Advocate | Spread | Note |
|-----------|-----------------|------------------|--------|------|
| Feasibility | 7/10 | ~3/10 (implied) | 4 | CA sees pattern already exists; DA sees hidden prerequisites |
| Blast Radius | 5/10 | 1/10 | 4 | CA counts files; DA counts shared state objects |
| Pattern Consistency | 8/10 | N/A | - | Both agree Pattern B (wall-clock threads) already exists |
| Reversibility | 8/10 | 2/10 | 6 | LARGEST DISAGREEMENT — CA sees toggleable hooks; DA sees once-threaded-always-threaded |
| Dependency Risk | 5/10 | 2/10 | 3 | Both agree convergence signal buffer is unprotected |
| Overall | 6/10 | 2/10 | 4 | |

### High Confidence (both agents converged independently)

1. **`self.signals` in ConvergenceAlerter has NO thread safety.** Both agents identified this independently. The `_alert_lock` protects deduplication only, not the primary signal buffer. This is the #1 prerequisite.

2. **CascadeTracker will crash instantly.** Plain dict `_active_chains` with three mutating operations and zero locks = `RuntimeError: dictionary changed size during iteration`. Guaranteed crash, not theoretical.

3. **FinnhubWebSocket proves the SAFE pattern — and it's the OPPOSITE of naive 50-thread injection.** FinnhubWebSocket deposits into a pending buffer; the main thread drains it. This is producer-only daemon + main-thread consumer. The proposal as originally conceived (50 threads writing directly to `record_signal()`) contradicts this proven pattern.

4. **InhabitantScheduler already exists for exactly this use case.** The Codebase Analyst found it: a heapq + threading.Lock + ThreadPoolExecutor scheduler purpose-built for wall-clock market intelligence dispatch.

5. **Thompson's `sample()` reads without acquiring `_lock`.** Both agents found this. Under current GIL, practically safe. Under Python 3.14 no-GIL (which MIDGE's runtime version supports), this becomes real data corruption.

6. **Bootstrap ordering is load-bearing.** The Thompson instance ordering bug (4 compounding bugs, fixed 2026-03-09) was caused by wrong initialization order. Threads starting before dependency injection = silent permanent bypass.

### The Recommended Approach: Producer-Consumer, Not Direct-Write

**The safe architecture is NOT "50 threads all writing to convergence alerter." It's "50 producer threads depositing to queues, one consumer thread draining into convergence alerter."**

This is exactly what FinnhubWebSocket already does. The migration path:

**Phase 0: Thread-Safety Prerequisites (MUST come first)**
- Add `threading.RLock` to ConvergenceAlerter protecting `self.signals`
- Add `threading.Lock` to CascadeTracker protecting `_active_chains`
- Add `threading.Lock` to OutcomeCollector protecting `_registered`
- Ensure `ThompsonSampler.sample()` acquires `_lock`
- Protect `ctx._cached_alerts[0]` handshake

**Phase 1: Producer-Consumer Signal Bus**
- Each data source daemon thread deposits signals to a `queue.Queue` (producer-only)
- One consumer thread (or the existing step loop) drains the queue and calls `record_signal()` + `check_convergence()`
- This preserves the ordering guarantee (sense before converge) while making sensing asynchronous

**Phase 2: Independent Timer Migration**
- Replace `ctx.model.add_step_hook(_market_sense_hook)` with `ctx.inhabitant_scheduler.register()` calls
- Each analytical subsystem (Granger, post-mortem, drift) gets its own wall-clock timer via InhabitantScheduler
- Mesa remains as heartbeat for bio systems

### What I Filtered Out

The Devil's Advocate's claim that "the bottleneck may not exist" deserves consideration but is LESS relevant than it appears. The user's request isn't about microsecond latency — it's about MIDGE feeling alive. 50 independent beings on their own clocks is an architectural vision, not a performance optimization. The council should not confuse "faster" with "living."

### Risks (validated)

1. **Thompson corruption during migration** — The feedback loop was just fixed. Any change to the signal flow path risks re-breaking it. Phase 0 locks mitigate this.
2. **Test suite cannot validate thread safety** — 4,536 deterministic tests won't catch timing bugs. New concurrency tests needed.
3. **GIL convoy effect at 50 threads** — Real concern for CPU-bound work (Granger, numpy). Mitigated by keeping CPU-heavy analytics on longer timers (not 50 competing threads).
4. **Windows daemon thread shutdown** — `wait=False` abandons threads mid-write. Need graceful shutdown protocol.

### Disagreements

- **Reversibility:** CA says 8/10 (hooks are toggleable), DA says 2/10 (threading changes are permanent). Truth is in between — Phase 0 locks are purely additive and beneficial regardless. Phase 1 queue pattern is reversible. Phase 2 scheduler migration is toggleable.
- **Whether the bottleneck exists:** DA argues API rate limits are the real constraint. True for throughput, but the user's request is about architecture (living ecosystem), not throughput. Both are correct in their framing.

### Next Step

Phase 0 (thread-safety locks) should be built NOW. It's beneficial regardless of whether the full migration happens — it fixes latent bugs that could manifest from existing threading (FinnhubWebSocket already writes via `_process_realtime_signals`). Phase 0 is pure risk reduction with zero downside.
